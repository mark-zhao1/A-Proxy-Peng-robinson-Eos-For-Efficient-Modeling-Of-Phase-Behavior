'''
Calculates phase stability and phase compositions
Use either ANN for the fugacity coefficient, or conventional EOS
Make sure that:
1) INPUTS sections match the fluid of interest
2) Set the path in Load models to where the ANN models are stored.
@author: markz
'''


import math
import numpy as np
from cmath import pi
import cProfile, pstats, io
from numba import jit
import os

# If use model
import pandas as pd
import pickle
import tensorflow as tf

# Debug
import matplotlib.pyplot as plt

class pr:
    def __init__(self):
        self.R = 8.31446261815324
        self.sqrt2 = 1.41421356237309504
        self.twosqrt2 = 2 * self.sqrt2
        self.onepsqrt2 = 2.41421356237309504
        self.onemsqrt2 = -0.41421356237309504
        self.useModel = False # By default


    ##########################################################################
    # Function definitions

    # Function bypassing problem with cubic root of small negative numbers
    def root3(self, num):
        if num < 0:
            return -(-num) ** (1. / 3.)
        else:
            return num ** (1. / 3.)

    # Returns K-values for all components using Wilson's correlation.
    def wilson_corr(self, Pr, Tr, w):
        #K = [1 / Pr[i] * math.exp(5.37 * (1 + w[i]) * (1 - 1 / Tr[i])) for i in range(len(w))]  # Truncated 5.373
        K = 1 / Pr * np.exp(5.37 * (1 + w) * (1 - 1 / Tr)) # Truncated 5.373
        # K = [1/Pr[i] * math.exp(5.373 * (1 + w[i]) * (1 - 1/Tr[i])) for i in range(len(w))]
        return K

    # Rachford-Rice
    # Get the phase compositions x, y from known K-values
    def objective_rr(self, beta, K, z):
        rrsum = ((K - 1) * z) / (1 + (K - 1) * beta)
        return np.sum(rrsum)

    # Analytical derivative of RR
    def rr_prime(self, beta, K, z):
        primesum = -1 * (z * (K - 1) ** 2) / (1 + beta * (K - 1)) ** 2
        return np.sum(primesum)

    # Solve RR equation for the vapor mole fraction beta
    #@profile
    def nr_beta(self, tol, K, beta, NRmaxit, z):
        Kmin = np.min(K)
        Kmax = np.max(K)

        # Michelsen's window
        xl = 1 / (1 - Kmax)
        xr = 1 / (1 - Kmin)
        xg = beta

        # Find root NR method
        check = 1
        i = 0
        while i < NRmaxit:
            i += 1
            y = np.sum(((K - 1) * z) / (1 + (K - 1) * xg))
            fp = np.sum(-1 * (z * (K - 1) ** 2) / (1 + xg * (K - 1)) ** 2) # This is the gradient at xg
            xn = xg - y / fp
            if xn < xl:
                xn = 0.5 * (xg + xl)
            if xn > xr:
                xn = 0.5 * (xg + xr)

            if xg != 0:
                check = abs(xn - xg)
                if check < tol:
                    break
                else:
                    xg = xn
            else:
                xg = xn

        if i > NRmaxit:
            print('Trouble in NR Solve')
            print('it = {}'.format(i))
            print('beta = {}'.format(xg))
            print('K: {}'.format(K))
            print('z: {}'.format(z))
        else:
             return xn, i


    # Analytical 3rd order polynomial solver. Modified to output sorted real roots only.
    def cubic_real_roots(self, p):
        # Input p = [A, B, C] such that x**3 + A*x**2 + B*x + C = 0
        q = (p[0]**2 - 3 * p[1]) / 9
        r = (2 * p[0]**3 - 9 * p[0] * p[1] + 27 * p[2]) / 54
        qcub = q**3
        d = qcub - r**2

        if abs(qcub) < 1E-16 and abs(d) < 1E-16:
            # 3 repeated real roots. Same as single root.
            #nroot=1
            z = np.array([-p[0] / 3])
            return z
        if abs(d) < 1E-16 or (d > 0 and abs(d) > 1E-16):
            # 3 distinct real roots
            #nroot = 3
            th = math.acos(r/math.sqrt(qcub))
            sqQ = math.sqrt(q)
            z = np.empty(3)
            z[0] = -2 * sqQ * math.cos(th/3) - p[0] / 3
            z[1] = -2 * sqQ * math.cos((th+2*pi)/3) - p[0] / 3
            z[2] = -2 * sqQ * math.cos((th+4*pi)/3) - p[0] / 3
            return z
        else:
            # 1 real root, 2 complex conjugates
            #nroots = 1
            e = self.root3(math.sqrt(-d) + abs(r))
            if r > 0:
                e = -e
            z = np.array([e + q/e - p[0]/3])
            return z

    def Z_roots_calc(self, a_mix_phase, b_mix_phase):
        A = a_mix_phase # Optimized: Already has Pr, Tr. R is cancelled.
        B = b_mix_phase
        p = [-(1 - B), (A - 3 * B ** 2 - 2 * B), -(A * B - B ** 2 - B ** 3)]
        Z_roots = self.cubic_real_roots(p)
        return Z_roots

    def Z_roots_det(self, a_mix_phase, b_mix_phase):
        '''
        :param a_mix_phase:
        :param b_mix_phase:
        :return: if multiple roots, return Z. Else, return False.
        '''
        A = a_mix_phase  # Optimized: Already has Pr, Tr. R is cancelled.
        B = b_mix_phase
        p = [-(1 - B), (A - 3 * B ** 2 - 2 * B), -(A * B - B ** 2 - B ** 3)]
        q = (p[0] ** 2 - 3 * p[1]) / 9
        r = (2 * p[0] ** 3 - 9 * p[0] * p[1] + 27 * p[2]) / 54
        qcub = q ** 3
        d = qcub - r ** 2

        if abs(d) < 1E-16 or (d > 0 and abs(d) > 1E-16):
            # 3 distinct real roots
            # nroot = 3
            th = math.acos(r / math.sqrt(qcub))
            sqQ = math.sqrt(q)
            Z = np.empty(3)
            Z[0] = -2 * sqQ * math.cos(th / 3) - p[0] / 3
            Z[1] = -2 * sqQ * math.cos((th + 2 * pi) / 3) - p[0] / 3
            Z[2] = -2 * sqQ * math.cos((th + 4 * pi) / 3) - p[0] / 3
            return Z
        else:
            return False # False = can use model

    def bm(self, phase_comps, b_i):
        return np.dot(phase_comps, b_i)

    def am(self, phase_comps, sum_xi_Aij):
        return np.dot(phase_comps, sum_xi_Aij)
    # Summation of a interactions, used in expression for lnphi

    def sum_a_interations(self, Nc, phase_comps, Am):
        sum_xi_Aij = np.zeros(Nc)
        for i in range(Nc):
            sum_xi_Aij[i] = np.dot(phase_comps, Am[i, :])
        return sum_xi_Aij

    @staticmethod
    @jit(nopython=True)
    def sum_a_interations_nb_static(Nc, phase_comps, Am):
        sum_xi_Aij = np.zeros(Nc)
        for i in range(Nc):
            sum_xi_Aij[i] = np.dot(phase_comps, Am[i, :])
        return sum_xi_Aij

    @staticmethod
    @jit(nopython=True)
    def am_nb_static(phase_comps, sum_xi_Aij):
        return np.dot(phase_comps, sum_xi_Aij)

    @staticmethod
    @jit(nopython=True)
    def bm_nb_static(phase_comps, b_i):
        return np.dot(phase_comps, b_i)



    def ln_phi_calc(self, b_i, a_mix, b_mix, sum_xjAij, Z):
        # Get fugacity coeff for each component in each phase.
        bibm = b_i / b_mix
        a1 = bibm * (Z - 1)
        a2 = - math.log(Z - b_mix)
        a3 = - 1 / (2.8284271247461903) * a_mix / b_mix
        a4a = sum_xjAij
        a4b = 2 / a_mix
        a4 = a4a * a4b - bibm
        a5 = math.log((Z + 2.414213562373095 * b_mix) / (Z + -0.4142135623730951 * b_mix))
        ln_phi = a1 + a2 + a3 * a4 * a5
        return ln_phi

    @staticmethod
    @jit(nopython=True)
    def ln_phi_calc_nbEOS(b_i, a_mix, b_mix, sum_xjAij, Z):
        # Get fugacity coeff for each component in each phase.
        bibm = b_i / b_mix

        a1 = bibm * (Z - 1)
        a2 = - math.log(Z - b_mix)
        #a3 = - 1 / (self.twosqrt2) * a_mix / b_mix
        a3 = - 1 / (2.8284271247461903) * a_mix / b_mix

        a4a = sum_xjAij
        a4b = 2 / a_mix
        #a4c = - b_i / b_mix
        a4 = a4a * a4b - bibm
        #a5 = math.log((Z + self.onepsqrt2 * b_mix) / (Z + self.onemsqrt2 * b_mix))
        a5 = math.log((Z + 2.414213562373095 * b_mix) / (Z + -0.4142135623730951 * b_mix))

        ln_phi = a1 + a2 + a3 * a4 * a5

        return ln_phi

    # Calculates mixing coefficients
    # Get a_mix_phase with mixing rule using a_i.
    def Vw(self, Nc,A,bip):
        Am = np.empty([Nc,Nc])
        for i in range(0,Nc):
            for j in range(0,Nc):
                Am[i,j] = np.sqrt(A[i] * A[j])*(1 - bip[i,j])
        return Am


    # Identify the phase stability result
    def caseid2(self, XX, itSSSAmax, TolXz, loop_count, sumXX, z):
        # Identify case
        tmp = abs(XX / z - 1)

        #tmp = [abs(XX[i] / z[i] - 1) for i in range(len(z))]
        if loop_count >= itSSSAmax:
            # Could not converge
            case_id = 1
        elif np.max(tmp) < TolXz:
            # Trivial case
            case_id = 2

        elif sumXX < 1+0.00023: # Accept stable phase if TPD > -3E-4
            # Converged, but G of x higher than G of z
            case_id = 3
        else:
            # Two phase is more stable
            case_id = -1
            # Debug
            #print('abs(XX/z-1): {}'.format(tmp))
            #print('sumXX: {}'.format(sumXX))

        return case_id

    def two_phase_flash_iterate(self, Pr, Tr, w, SSmaxit, SStol, TolRR, Nc, Am, b_i, NRmaxit,z):
        # Wilson's Correlation for K values
        K = self.wilson_corr(Pr, Tr, w) # If remove will have local variable clash with global

        # Initial vapor phase mole fraction
        beta = 0.5

        ###################################
        # PROFILING
        # Single iteration of SS in two-phase flash.
        #for _ in range(100000):
        #    self.two_phase_flash_SS_test(Nc, K, flag, outer_loop_count, TolRR, b_i, Am, z)
        #return 'profiling', 'profiling'
        ###################################

        # If using ANNs, declare global var X_unprepared to pass arguments to ANN functions.
        if self.useModel:
            global X_unp
            X_unp = np.empty((Nc, 4))

        # Outer loop start
        # Increment flag to break while loop.
        flag = 0
        outer_loop_count = 0

        while outer_loop_count < SSmaxit and flag < 1:  # Flag exit condition at 1 to print converged+1 x, y, K-values
            outer_loop_count += 1

            # Call NR method for beta (vapor fraction)
            beta, i_count = self.nr_beta(TolRR, K, beta, NRmaxit, z)

            # Get phase compositions from K and beta
            x = z / (1 + beta * (K - 1))
            y = K * x

            # Normalize
            x = x / np.sum(x)
            y = y / np.sum(y)

            # Check material balance for each component
            for comp in range(len(z)):
                if abs(z[comp] - (x[comp] * (1 - beta) + y[comp] * beta)) > 1E-6:# 1E-10 for EOS
                    print('Caution: Material balance problem for component ' + str(comp))
                    # debug
                    print(abs(z[comp] - (x[comp] * (1 - beta) + y[comp] * beta)))

            # Check mole fractions
            if 1 - np.sum(x) > 1E-12 or 1 - np.sum(y) > 1E-12:
                print('''Caution: Phase comp don't add up to 1.''')

            #print('Liquid comp: ' + str(x))
            #print('Vapor comp: ' + str(y))

            #####################################################
            # Liquid
            # Get parameters for Peng-Robinson EOS that are composition dependent.
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            # If using ANNs, determine if the EOS has a single root. Skip this if assume single root at all conditions.
            # Otherwise, calculate the EOS roots.
            if self.useModel:
                Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
            else:
                Z = self.Z_roots_calc(a_mix, b_mix)

            # If ANNs are used, and only one root exists, use the ANNs for the fugacity coefficient.
            # Else, use conventional EOS.
            if type(Z) == bool:
                X_unp[:, :2] = a_mix, b_mix
                X_unp[:, 2:] = np.column_stack((b_i, sum_xiAij))
                ln_phi_x = ANN_numba_noargs()
            else:
                # Use EOS lnphi
                if len(Z) > 1 and min(Z) > 0:
                    print('SA: More than 1 root. Gibb\'s minimization performed.')
                    ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                else:
                    ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))

            ######################################################
            # Vapor
            sum_xiAij = self.sum_a_interations(Nc, y, Am)
            a_mix = self.am(y, sum_xiAij)
            b_mix = self.bm(y, b_i)

            # If using ANNs, determine if the EOS has a single root. Skip this if assume single root at all conditions.
            # Otherwise, calculate the EOS roots.
            if self.useModel:
                Z = self.Z_roots_det(a_mix, b_mix)
            else:
                Z = self.Z_roots_calc(a_mix, b_mix)

            # If ANNs are used, and only one root exists, use the ANNs for the fugacity coefficient.
            # Else, use conventional EOS.
            if type(Z) == bool:
                X_unp[:, :2] = a_mix, b_mix
                X_unp[:, 2:] = np.column_stack((b_i, sum_xiAij))
                ln_phi_y = ANN_numba_noargs()
            else:
                # Use EOS lnphi
                if len(Z) > 1 and min(Z) > 0:
                    print('SA: More than 1 root. Gibb\'s minimization performed.')
                    ln_phi_y, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, y)
                else:
                    ln_phi_y = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))

            # Converge check
            ln_phi_diff = ln_phi_x - ln_phi_y
            c = np.abs(ln_phi_diff - np.log(K))
            if np.max(c) < SStol:
                flag += 1
                #print('Exit flag:' + str(flag))

            # Update K
            #print('K old: ' + str(K))
            K = np.exp(ln_phi_diff)
            #print('K new: ' + str(K))
            #print('########################################')
        print('Flash SS iterations: {}'.format(outer_loop_count))
        # Compute d, for use in 3 phase SA.
        d = ln_phi_x + np.log(x)
        return x, y, d

    def kappa(self, w):
        kappa = []  # Verified
        for comp in range(len(w)):
            if w[comp] <= 0.49:
                kappa.append(0.37464 + 1.54226 * w[comp] - 0.26992 * w[comp] ** 2)
            else:
                kappa.append(0.37964 + w[comp] * (1.48503 + w[comp] * (-0.164423 + w[comp] * 0.016666)))
        return np.array(kappa)

    def aibi(self, P,T,w,Pr,Tr,Pc,Tc):
        PT2 = P / T ** 2
        #PT = P / T
        Kappa = self.kappa(w)
        alpha = (1 + Kappa * (1 - np.sqrt(Tr))) ** 2
        a_i = 0.457236 * alpha * Tc ** 2 * PT2 / Pc
        b_i = 0.0778 * Pr / Tr  # Optimized Bi. Tr, Pr, removed R
        return a_i, b_i

    # Outputs lower Gibbs root and corresponding ln_phi
    def checkG(self, b_i, a_mix, b_mix, sum_xiAij, Z, x):
        Zmin = min(Z)
        Zmax = max(Z)
        if Zmin < 0:
            ln_phi_max = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmax)
            return ln_phi_max, Zmax

        ln_phi_min = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmin)
        ln_phi_max = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmax)

        arr = x * (ln_phi_min - ln_phi_max)
        if np.sum(arr) > 0:
            return ln_phi_max, Zmax
        else:
            return ln_phi_min, Zmin

    # For profiling
    def do_cprofile(self, func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            repeats = int(1E6)
            try:
                for _ in range(repeats):
                    profile.enable()
                    result = func(*args, **kwargs)
                    profile.disable()
                return result
            finally:
                s = io.StringIO()
                sortby = 'tottime'
                ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())
                print('Profiled  %d repeats. Divide by that number for per iteration times.' % (repeats))
        return profiled_func

    # SA SS single iteration standalone, for profiling only.
    # Constant variables
    #@profile
    def SA_SS_single_it(self, Nc, x, b_i, Am, XX, d, tolSSSA, exit_flag):
        sum_xiAij = self.sum_a_interations_nb_static(Nc, x, Am)
        b_mix = self.bm_nb_static(x, b_i)
        a_mix = self.am_nb_static(x, sum_xiAij)

        # Prepare variables for ANN
        global X_prepared_nC4
        global X_prepared_nC10
        X_prepared_nC4, X_prepared_nC10 = X_predict_nb(a_mix, b_mix, b_i, sum_xiAij)

        # If using ANNs, determine if the EOS has a single root. Skip this if assume single root at all conditions.
        # Otherwise, calculate the EOS roots.
        if self.useModel:
            Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
        else:
            Z = Z_roots_calc(a_mix, b_mix)

        if type(Z) == bool:
            # Use npnbANN ln_phi
            y_hat_nC4, y_hat_nC10 = ANN_numba_noargs()
            ln_phi_x = np.array([y_hat_nC4[0], y_hat_nC10[0]])
        else:
            # Use EOS lnphi
            if len(Z) > 1 and min(Z) > 0:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
            else:
                ln_phi_x = self.ln_phi_calc_nbEOS(b_i, a_mix, b_mix, sum_xiAij, max(Z))

        # Compute convergence
        tmp = np.abs(ln_phi_x + np.log(XX) - d)

        # Compute convergence nb
        exit_flag = convergence_check_SSSA_nb(ln_phi_x, XX, d, exit_flag, tolSSSA)
        # Update XX and x
        XX, x = update_SSSA_nb(ln_phi_x, d)


        '''# Update XX
        XX = np.exp(d - ln_phi_x)

        # Update x
        sumXX = np.sum(XX)
        x = XX / sumXX

        # Check convergence
        if np.max(tmp) < tolSSSA:
            exit_flag += 1'''
        if exit_flag > 1:
            loop_count = it
            #break
        return

    # Two-phase flash SS single iteration standalone, for profiling only.
    #@profile
    def two_phase_flash_SS_test(self, Nc, K, flag, outer_loop_count, TolRR, b_i, Am, z):
        beta = 0.5
        while outer_loop_count < SSmaxit and flag < 2:  # Flag exit condition at 2 to print converged+1 x, y, K-values
            outer_loop_count += 1

            # Call NR method for beta (vapor fraction)
            beta, i_count = self.nr_beta(TolRR, K, beta, NRmaxit, z)

            # Get Phase compositions from K and beta
            x = z / (1 + beta * (K - 1))
            y = K * x

            # Normalize
            x = x / np.sum(x)
            y = y / np.sum(y)

            #####################################################
            # Liquid
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            # All EOS variables defined, solve EOS for each phase
            Z = self.Z_roots_calc(a_mix, b_mix)

            # If more than one root, select the root with lower Gibbs energy.
            if len(Z) > 1:
                ln_phi_x, Z = self.checkG(Nc, b_i, a_mix, b_mix, sum_xiAij, Z, x)
            else:
                ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            ######################################################
            # Vapor
            sum_xiAij = self.sum_a_interations(Nc, y, Am)
            a_mix = self.am(y, sum_xiAij)
            b_mix = self.bm(y, b_i)

            # All EOS variables defined, solve EOS for each phase
            Z = self.Z_roots_calc(a_mix, b_mix)

            # If more than one root, select the root with lower Gibbs energy.
            if len(Z) > 1:
                ln_phi_y, Z = self.checkG(Nc, b_i, a_mix, b_mix, sum_xiAij, Z, y)
            else:
                ln_phi_y = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

            # Converge check
            ln_phi_diff = ln_phi_x - ln_phi_y
            c = np.abs(ln_phi_diff - np.log(K))
            if np.max(c) < SStol:
                flag += 1
            else:
                flag = 0

            # Update K
            K = np.exp(ln_phi_diff)
        return

    #@do_cprofile
    def stability_analysis(self, T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc, K, TolXz):
        # Declare global variable if using ANns. global var X_unprepared passes arguments to ANN functions.
        if self.useModel:
            global X_unp
            X_unp = np.empty((Nc, 4))

        # Get parameters for Peng-Robinson EOS which are composition dependent.
        sum_xiAij = self.sum_a_interations(Nc, z, Am)
        a_mix = self.am(z, sum_xiAij)
        b_mix = self.bm(z, b_i)


        if self.useModelSA:
            Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
        else:
            Z = self.Z_roots_calc(a_mix, b_mix)

        # If ANNs are used, and only one root exists, use the ANNs for the fugacity coefficient.
        # Else, use conventional EOS.
        if type(Z) == bool:
            # Use ANN lnphi
            # Construct the independent var to pass to ANNs
            X_unp[:, :2] = a_mix, b_mix
            X_unp[:, 2:] = np.column_stack((b_i, sum_xiAij))
            ln_phi_z = ANN_numba_noargs()
        else:
            # Use EOS lnphi
            if len(Z) > 1 and min(Z) > 0:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_z, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
            else:
                ln_phi_z = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))

        d = ln_phi_z + np.log(z)
        #################
        # Liquid-like search for instability
        XX = z / K
        x = XX / np.sum(XX) # Maybe define sumXX beforehand
        # SS in SA
        exit_flag = 0

        '''
        # PROFILING
        # Objective: Run the first iteration outside of the speed (warm up).
        # Warm up the first iteration in the context of speed timing, since the first iteration is considerably slower
        # compared to subsequent iterations.
        # Uncomment the block corresponding to the method selected.
        
        # Warm up
        # If using ANNs, declare global var X_unprepared to pass arguments to ANN functions.
        global X_unp # For X_unprepared
        X_unp = np.empty((Nc,4))
        X_unp[:,:2] = a_mix, b_mix
        X_unp[:,2:] = np.column_stack((b_i, sum_xiAij))
        ANN_numba_noargs()

        # Warm up regular EOS nb
        sum_xiAij = self.sum_a_interations(Nc, x, Am)
        b_mix = self.bm(x, b_i)
        a_mix = self.am(x, sum_xiAij)
        Z = self.Z_roots_calc(a_mix, b_mix)
        ln_phi_x = self.ln_phi_calc_nbEOS(b_i, a_mix, b_mix, sum_xiAij, max(Z))

        # Warm up npbn for EOS
        Z = Z_roots_calc(a_mix, b_mix)

        # Warm up mixing properties nb
        sum_xiAij = self.sum_a_interations_nb_static(Nc, x, Am)
        b_mix = self.bm_nb_static(x, b_i)
        a_mix = self.am_nb_static(x, sum_xiAij)
        X_prepared_nC4, X_prepared_nC10 = X_predict_nb(a_mix, b_mix, b_i, sum_xiAij)

        # Warm up convergence check and SSSA update
        exit_flag2 = convergence_check_SSSA_nb(ln_phi_x, XX, d, exit_flag, tolSSSA)
        # Update XX and x
        XX2, x2 = update_SSSA_nb(ln_phi_x, d)

        # For profiling. Single iteration of SS in SA.
        for _ in range(1000):
            self.SA_SS_single_it(Nc, x, b_i, Am, XX, d, tolSSSA, exit_flag)
        return

        ###############'''

        # DEBUG: Iterations counter
        self.liq_max_tmp = []
        self.liq_it = []

        # If using ANNs, declare global var X_unprepared to pass arguments to ANN functions.
        X_unp = np.empty((Nc, 4))

        # Start loop for stationarity point search
        for loop_count in range(int(itSSSAmax+1)):
            # Get parameters for Peng-Robinson EOS that are composition dependent.
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            # If ANNs are used, and only one root exists, use the ANNs for the fugacity coefficient.
            # Else, use conventional EOS.
            if self.useModelSA:
                Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
            else:
                Z = self.Z_roots_calc(a_mix, b_mix)
            if type(Z) == bool:
                # Use ANN lnphi
                # Construct the independent var to pass to ANNs
                X_unp[:, :2] = a_mix, b_mix
                X_unp[:, 2:] = np.column_stack((b_i, sum_xiAij))
                ln_phi_x = ANN_numba_noargs()
            else:
                # Use EOS lnphi
                if len(Z) > 1 and min(Z) > 0:
                    print('SA: More than 1 root. Gibb\'s minimization performed.')
                    ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                else:
                    ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))

            # Compute convergence by checking stationarity
            tmp = np.abs(ln_phi_x + np.log(XX) - d)
            #print('In SA Liquid-Like: Tmp = {}'.format(tmp))
            # Log tmp (debug)
            self.liq_max_tmp.append(max(tmp))
            self.liq_it.append(loop_count)

            # Update XX
            XX = np.exp(d - ln_phi_x)

            # Update x
            sumXX = np.sum(XX)
            x = XX / sumXX

            # Check convergence
            if np.max(tmp) < tolSSSA:
                break

        sumXX_list = np.empty(2)
        sumXX_list[0] = sumXX
        liq_case = self.caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX, z)
        # DEBUG
        #print('liq loop_count: {}'.format(loop_count))
        #print('DEBUG: Liq trivial stationary point check: {}'.format(max(abs(XX / z - 1))))
        #################

        # Vapor-like search for instability
        if liq_case > 0:
            XX = z * K
            x = XX / np.sum(XX)
            exit_flag = 0

            # DEBUG: Iteration counter
            self.vap_max_tmp = []
            self.vap_it = []
            for loop_count in range(int(itSSSAmax+1)):
                #a_mix, b_mix = ambm(x, b_i, Am)
                sum_xiAij = self.sum_a_interations(Nc, x, Am)
                a_mix = self.am(x, sum_xiAij)
                b_mix = self.bm(x, b_i)

                # If ANNs are used, and only one root exists, use the ANNs for the fugacity coefficient.
                # Else, use conventional EOS.
                if self.useModelSA:
                    Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
                else:
                    Z = self.Z_roots_calc(a_mix, b_mix)
                if type(Z) == bool:
                    # Use ANN lnphi
                    # Construct the independent var to pass to ANNs
                    X_unp[:, :2] = a_mix, b_mix
                    X_unp[:, 2:] = np.column_stack((b_i, sum_xiAij))
                    ln_phi_x = ANN_numba_noargs()
                else:
                    # Use EOS lnphi
                    if len(Z) > 1 and min(Z) > 0:
                        print('SA: More than 1 root. Gibb\'s minimization performed.')
                        ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                    else:
                        ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))

                # Compute convergence
                tmp = np.abs(ln_phi_x + np.log(XX) - d)
                # Log tmp (debug)
                self.vap_max_tmp.append(max(tmp))
                self.vap_it.append(loop_count)

                # Update XX
                XX = np.exp(d - ln_phi_x)

                # Update x
                sumXX = np.sum(XX)
                x = XX / sumXX

                # DEBUG
                #print(loop_count)

                # Check convergence
                if np.max(tmp) < tolSSSA:
                    break

            sumXX_list[1] = sumXX
            vap_case = self.caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX, z)
            #print('DEBUG: Vap trivial stationary point check: {}'.format(max(abs(XX / z - 1))))
            #print('vap loop_count: {}'.format(loop_count))
            #print('Vapor and liquid stationarity equation residuals')
            #print(self.vap_tmp[-1], self.liq_tmp[-1])
        else:
            vap_case = 0
        return sumXX_list, liq_case, vap_case

    def ini3phrsa(self, Nc, x, y, itrial, d):
        '''
        Returns composition of guess phase for 3-phase SA, depending on itrial.

        :param nc: number of components
        :param x: mole fraction of liquid phase
        :param y: mole fraction of vapor phase
        :param itrial: identifying for SA trial. 1: 0.5*(x+y), 2: 0.999 CO2, 3: 0.999 Heaviest C, 4: exp(d)
        :param d: ln x + ln phi of one of the 2 phases in equilibrium
        :return:
        '''

        if itrial == 1:
            XX = 0.5 * (x + y)
        elif itrial == 2:
            diff = 0.001 / (Nc - 1)
            XX = np.ones(Nc) * diff
            XX[0] = 0.999
        elif itrial == 3:
            diff = 0.001 / (Nc - 1)
            XX = np.ones(Nc) * diff
            XX[-1] = 0.999
        elif itrial == 4:
            XX = np.exp(d)
        return XX

    def caseid3(self, XX, itSSSAmax, TolXz, TolsumXX, loop_count, sumXX, SAz, SAnz):
        '''
        Identifies the case in stability analysis after a stationarity point is search.
        The cases are as follows:
        case_id==1; Could not converge, exceed max iteration count
        case_id==2; Trivial case, stationarity point is the same as the reference phase.
        case_id==3; Converged, but G of x is higher than the reference phase. Single phase in this case
        case_id==-1; Converged, found a phase composition with G lower than reference phase.

        :param XX:          Phase composition of trial phase
        :param itSSSAmax:   Max number of iterations
        :param TolXz:       Tolerance for trivial case
        :param TolsumXX:    Tolerance for stable phase, use for ANNs only.
        :param loop_count:  Loop count of stationarity point search
        :param sumXX:       Sum of compositions of trial phase
        :param SAz:         Composition of phase x
        :param SAnz:        Composition of pahse y
        :return:            case_id
        '''
        # Identify case
        tmp1 = abs(XX / SAz - 1)
        tmp2 = abs(XX / SAnz - 1)

        # tmp = [abs(XX[i] / z[i] - 1) for i in range(len(z))]
        if loop_count >= itSSSAmax:
            # Could not converge
            case_id = 1
        elif np.max(tmp1) < TolXz or np.max(tmp2) < TolXz:
            # Trivial case
            case_id = 2

        elif sumXX < 1+TolsumXX: # Add tolerance for stable phase
            # Converged, but G of x higher than G of z
            case_id = 3
        else:
            # Two phase is more stable
            case_id = -1
            # Debug
            # print('abs(XX/z-1): {}'.format(tmp))
            # print('sumXX: {}'.format(sumXX))

        return case_id

    def stationarity_SA_3(self, Nc, Am, b_i, tolSSSA, itSSSAmax, XX, d):
        '''
        Finds a stationarity point given a starting guess trial phase composition and reference phase G

        :param tolSSSA:
        :param itSSSAmax:
        :param z:           Overall composition of test phase. In 3 phase SA, this is the composition of one of the 2 phases.
        :param XX:          Composition of the trial phase
        :param d:           ln_phi_x + np.log(x) from one of the 2 phase flash phases.
        :return:
        '''

        x = XX / np.sum(XX)

        # SS in SA
        exit_flag = 0
        old = 0
        count = 0

        for loop_count in range(int(itSSSAmax + 2)):
            # a_mix, b_mix = ambm(x, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            if self.useModel:
                Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
            else:
                Z = self.Z_roots_calc(a_mix, b_mix)

            if type(Z) == bool:
                # Use ANN lnphi
                # Construct the independent var to pass to ANNs
                X_unp[:, :2] = a_mix, b_mix
                X_unp[:, 2:] = np.column_stack((b_i, sum_xiAij))
                ln_phi_x = ANN_numba_noargs()
            else:
                # Use EOS lnphi
                if len(Z) > 1 and min(Z) > 0:
                    print('SA: More than 1 root. Gibb\'s minimization performed.')
                    ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                else:
                    ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))

            # Compute convergence to stationarity point
            tmp = np.abs(ln_phi_x + np.log(XX) - d)

            # Update XX
            XX = np.exp(d - ln_phi_x)

            # Update x
            sumXX = np.sum(XX)
            x = XX / sumXX

            # Check convergence
            if np.max(tmp) < tolSSSA:
                exit_flag += 1
            if exit_flag > 1:
                break
            # Break if same for 5 iterations in a row
            if np.max(tmp) == old:
                count += 1
            if count > 4:
                print('DEBUG Break from stationarity_SA_3: Same np.max(tmp) for 5 iterations in a row.')
                print('DEBUG np.max(tmp): {}'.format(old))
                break
            old = np.max(tmp)

            # Debug
            if loop_count > itSSSAmax:
                print('DEBUG Max loop_count exceeded in stationarity_SA_3. Loop_count: {}'.format(loop_count))
                print('DEBUG np.max(tmp): {}'.format(old))
                break
        return XX, sumXX, loop_count

    def load_pipeline_values(self, pipelinePath):
        '''
        # Extract min, range values of transformation pipelines.
        :param pipelinePath: List of paths to the pipeline .pkl files
        :return:
        '''
        # Load transformation pipelines
        self.attr_full_pipelines = []
        self.label_full_pipelines = []
        for i in pipelinePath:
            with open(i, 'rb') as f:
                temp = pickle.load(f)
                self.attr_full_pipelines.append(temp)
                temp = pickle.load(f)   # Loads the next variable (label pipeline)
                self.label_full_pipelines.append(temp)

        global min_attr
        global range_attr
        global min_label
        global range_label

        min_attr = np.array([i.named_transformers_.num.named_steps.min_max_scaler.data_min_
                    for i in self.attr_full_pipelines],dtype=np.float32)
        range_attr = np.array([i.named_transformers_.num.named_steps.min_max_scaler.data_range_
                      for i in self.attr_full_pipelines],dtype=np.float32)
        min_label = np.array([i[0].data_min_ for i in self.label_full_pipelines],dtype=np.float32)
        range_label = np.array([i[0].data_range_ for i in self.label_full_pipelines],dtype=np.float32)

        print('Transformation pipeline values extracted.')
        return



    def load_wb(self, wbPath):
        '''
        Load weights and biases into global namespace
        :param wbPath: Contains all .npz weights and bias files for the fluid.
        :return:
        '''

        global weight
        global bias

        weight = np.array([])
        bias = np.array([])
        for i in os.listdir(wbPath):
            temp_w, temp_b = np.load(os.path.join(wbPath, i), allow_pickle=True)['wb']
            weight = np.append(weight, temp_w)
            bias = np.append(bias, temp_b)
        return


@jit(nopython=True)
def convergence_check_SSSA_nb(ln_phi_x, XX, d, flag, tolSSSA):
    tmp = np.max(np.abs(ln_phi_x + np.log(XX) - d))
    if tmp < tolSSSA:
        flag += 1
    return flag


@jit(nopython=True)
def update_SSSA_nb(ln_phi_x, d):
    XX = np.exp(d - ln_phi_x)
    sumXX = np.sum(XX)
    x = XX / sumXX
    return XX, x


@jit(nopython=True)
def X_predict_nb(a_mix, b_mix, b_i, sumxjAij):
    return np.array([a_mix, b_mix, b_i[0], sumxjAij[0]]), np.array([a_mix, b_mix, b_i[1], sumxjAij[1]])


# Analytical 3rd order polynomial solver. Modified to output sorted real roots only.
@jit(nopython=True)
def cubic_real_roots(p):
    # Input p = [A, B, C] such that x**3 + A*x**2 + B*x + C = 0

    q = (p[0]**2 - 3 * p[1]) / 9
    r = (2 * p[0]**3 - 9 * p[0] * p[1] + 27 * p[2]) / 54
    qcub = q**3
    d = qcub - r**2

    if abs(qcub) < 1E-16 and abs(d) < 1E-16:
        # 3 repeated real roots. Same as single root.
        #nroot=1
        z = np.array([-p[0] / 3])
        return z
    if abs(d) < 1E-16 or (d > 0 and abs(d) > 1E-16):
        # 3 distinct real roots
        #nroot = 3
        th = math.acos(r/math.sqrt(qcub))
        sqQ = math.sqrt(q)
        z = np.empty(3)
        z[0] = -2 * sqQ * math.cos(th/3) - p[0] / 3
        z[1] = -2 * sqQ * math.cos((th+2*pi)/3) - p[0] / 3
        z[2] = -2 * sqQ * math.cos((th+4*pi)/3) - p[0] / 3
        return z
    else:
        # 1 real root, 2 complex conjugates
        #nroots = 1
        e = root3(math.sqrt(-d) + abs(r))
        if r > 0:
            e = -e
        z = np.array([e + q/e - p[0]/3])
        return z

@jit(nopython=True)
def Z_roots_calc(a_mix_phase, b_mix_phase):
    A = a_mix_phase # Optimized: Already has Pr, Tr. R is cancelled.
    B = b_mix_phase
    p = [-(1 - B), (A - 3 * B ** 2 - 2 * B), -(A * B - B ** 2 - B ** 3)]
    Z_roots = cubic_real_roots(p)
    return Z_roots

# Function bypassing problem with cubic root of small negative numbers
@jit(nopython=True)
def root3(num):
    if num < 0:
        return -(-num) ** (1. / 3.)
    else:
        return num ** (1. / 3.)

''' ANN prediction. 
All variables loaded in global. 
Normalization and inverse-normalization done within.
All components calculated together
Use j as counter to separate components. Component weight and biases are saved together in w and b
Must declare types for each variable. Numba can't infer them all.
np.dot in Numba requires all the same dtype
'''
#@jit(nopython=True)
def ANN_numba_noargs():
    # Set independent variable from global namespace
    x = X_unp.astype(np.float32)
    # Declare variables. Required for compilation
    y_hat = np.empty(6,dtype=np.float32)

    for i in range(0, 6):
        j = 5 * i
        xx = (x[i] - min_attr[i]) / range_attr[i]
        l0 = np.dot(xx, weight[j]) + bias[j]
        l0 = np.where(l0 > 0, l0, l0 * 0.1)
        l1 = np.dot(l0, weight[j + 1]) + bias[j + 1]
        l1 = np.where(l1 > 0, l1, l1 * 0.1)
        l2 = np.dot(l1, weight[j + 2]) + bias[j + 2]
        l2 = np.where(l2 > 0, l2, l2 * 0.1)
        l3 = np.dot(l2, weight[j + 3]) + bias[j + 3]
        l3 = np.where(l3 > 0, l3, l3 * 0.1)
        l4 = np.dot(l3, weight[j + 4]) + bias[j + 4]
        y_hat[i] = range_label[i] * l4 + min_label[i]
    return y_hat



if __name__ == "__main__":
    ########################################################################################
    # INPUTS
    # Pressure [psia]
    P = 1189.8431166429973
    # Temperature [degR]
    T = 549.67
    # Injection Gas Mole Ratio
    gas_r = 0.7055172413793104

    # Monahans Clearfork Oil
    # Molar composition of oil
    z = np.array([0.0001, 0.3056, 0.2027, 0.1589, 0.2327, 0.1000])
    # Molar composition of injection gas
    z_gas = np.array([0.95, 0.05, 0.0000, 0.0000, 0.0000, 0.0000])
    # Overall molar composition
    z = gas_r*z_gas + (1-gas_r)*z
    # Acentric factor (omega)
    w = np.array([0.225, 0.008, 0.127, 0.240, 0.609, 1.042])
    # Critical Pressure [psia]
    Pc = np.array([1069.87, 667.20, 658.59, 487.51, 329.42, 258.78])
    # Critical Temperature [degR]
    Tc = np.array([547.56, 343.08, 612.02, 835.06, 1086.35, 1444.93])
    # Binary Interaction Parameters
    BIP = np.array([[0.000, 0.094, 0.094, 0.094, 0.095, 0.095],
                    [0.094, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.094, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.094, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.095, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.095, 0.000, 0.000, 0.000, 0.000, 0.000]])
    # Tolerance for Newton-Raphson convergence, in constant-K flash
    NRtol = 1E-12
    # Maximum iteration count for Newton-Raphson convergence, in constant-K flash
    NRmaxit = 100
    # Tolerance for residuals of the fugacity equation, in successive substitution flash
    SStol = 5E-6  #1E-10 for EOS, 1E-6 for ANN. Higher to be safe
    # Tolerance for residuals of the stationarity equation, in successive substitution stability analysis
    tolSSSA = 1E-5 #1E-5 for EOS # EOS 1E-10
    # Maximum iteration count for successive substitution
    SSmaxit = 1E6 # Lower this to identify convergence problems faster.
    # Tolerance for Rachford-Rice iterations
    TolRR = 1E-10
    # Tolerance to identify the trivial case in stability analysis
    TolXz = 1E-5 #1E-8 for EOS, 1E-5 for ANN
    # Tolerance to identify an stable case in stability analysis
    TolsumXX = 0.00023 # 0 for EOS, 0.00023 for ANN
    # Maximum iteration count for successive substitution stability analysis
    itSSSAmax = 1E6 # 1E6 for EOS

    # More global constants
    # Reduced temperature
    Tr = T / Tc
    # Reduced pressure
    Pr = P / Pc
    # Number of components
    Nc = len(z)

    phase_num = 1
    row_index = 0

    #####################################################################################
    # Instantiate class
    pr = pr()

    # Use Model?
    pr.useModel = True
    pr.useModelSA = True

    # Load models
    # Specify raw path to the models.
    path = r'C:\Users\markz\OneDrive\Documents\logs\Monahans_with_mixing_line\logs\scalars'
    dirList = [i for i in os.listdir(path) if '_Monahans' in i]
    modelPath = [os.path.join(path, i) for i in dirList]
    pipelinePath = [os.path.join(i, j) for i in modelPath for j in os.listdir(i) if '.pkl' in j]
    wbPath = r'C:\Users\markz\OneDrive\Documents\logs\Monahans_with_mixing_line\logs\scalars\Monahans_wbfiles'

    # Load transformation pipeline values to global namespace
    pr.load_pipeline_values(pipelinePath)

    # Load npANN weights and biases to global namespace
    pr.load_wb(wbPath)

    # Parameters independent of composition placed out of loop.
    # Used in either stability analysis or 2-phase PT flash.

    # Get all K-values from Wilson
    K = pr.wilson_corr(Pr, Tr, w)
    ln_K = np.log(K)

    # Get all ai, bi values
    a_i, b_i = pr.aibi(P, T, w, Pr, Tr, Pc, Tc)

    # Get Vw mixing, part with BIPs and square roots
    Am = pr.Vw(Nc,a_i,BIP)
    ##########################################################################################
    # Debug
    pr.tmp_list = []
    pr.z_list = []

    # Stability Analysis
    sumXX_list, liq_case, vap_case = pr.stability_analysis(T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc, K, TolXz)

    # Get TPD
    TPD = -math.log(max(sumXX_list))
    print('TPD: {}'.format(TPD))
    print(sumXX_list)

    print('At P = %s bar, T = %s K, gas_r = %s' % (P, T, gas_r))
    if liq_case < 0 or vap_case < 0:
        print('Single phase unstable, TPD = %s' % TPD)
        print('Run 2-phase flash.')

        phase_num = 2
        # Now call 2-phase flash func. Return only converged composition. Optimize by re-using calculated
        # variables.

        x, y, d = pr.two_phase_flash_iterate(Pr, Tr, w, SSmaxit, SStol, TolRR, Nc, Am, b_i, NRmaxit, z)
        print('liq and vap comp:')
        print(x, y)

        # Do 3 phase SA
        itrial = 1
        x_trial = pr.ini3phrsa(Nc, x, y, itrial, d)
        # Call 3 phase SA stationary point search
        XX, sumXX, loop_count = pr.stationarity_SA_3(Nc, Am, b_i, tolSSSA, itSSSAmax, x_trial, d)
        # Call 3 phase case identification
        caseid = pr.caseid3(XX, itSSSAmax, TolXz, TolsumXX, loop_count, sumXX, x, y)
        if caseid > 0:
            itrial = 2
            x_trial = pr.ini3phrsa(Nc, x, y, itrial, d)
            XX, sumXX, loop_count = pr.stationarity_SA_3(Nc, Am, b_i, tolSSSA, itSSSAmax, x_trial, d)
            caseid = pr.caseid3(XX, itSSSAmax, TolXz, TolsumXX, loop_count, sumXX, x, y)
            if caseid > 0:
                itrial = 3
                x_trial = pr.ini3phrsa(Nc, x, y, itrial, d)
                XX, sumXX, loop_count = pr.stationarity_SA_3(Nc, Am, b_i, tolSSSA, itSSSAmax, x_trial, d)
                caseid = pr.caseid3(XX, itSSSAmax, TolXz, TolsumXX, loop_count, sumXX, x, y)
                if caseid > 0:
                    itrial = 4
                    x_trial = pr.ini3phrsa(Nc, x, y, itrial, d)
                    XX, sumXX, loop_count = pr.stationarity_SA_3(Nc, Am, b_i, tolSSSA, itSSSAmax, x_trial, d)
                    caseid = pr.caseid3(XX, itSSSAmax, TolXz, TolsumXX, loop_count, sumXX, x, y)
        if caseid < 0:
            # Call 3 phase flash
            print('3 phase')
        else:
            print('2 phase stable')


    elif liq_case > 0 and vap_case > 0:
        print('Single phase stable')
        print('P = %s bar, T = %s K' % (P, T))
        print('Liq case: %d, Vap case: %d' % (liq_case, vap_case))
        # Copy single phase composition

    print('END')

