# -*- coding: utf-8 -*-


import sympy as sp


# Load the masses of hadrons
from hadrons_mass import *


def Gdr(EE, m1, m2, Λ): # two body propagators
    μ = m1*m2 / (m1+m2)
    k = sp.sqrt(2*μ*(EE-m1-m2) )
    erfi = sp.erf(sp.sqrt(2)*k/Λ)
    f = sp.exp(-2*k**2 / Λ**2)
    return -μ*Λ / (2*sp.pi)**(3/2) + μ*k/2/sp.pi * f * (erfi-1j )

def Kallen(x, y, z): # Kallen function
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*x*z


# B+ -> D+ D- K+
# 0+ channel
def channel0p_Bp(EE, Λ, A, B, r0, C11, C10, E11, E10, ith, jth, m1=Dstpm, m2=Kstpm, m3=Dst0, m4=Kst0):
    UBp = sp.Matrix([[A*r0, -B*r0]]) / sp.sqrt(2)

    V0p = sp.Matrix([[C11+C10, C11-C10], 
                    [C11-C10, C11+C10]]) / 2

    # propagators for D^*- K^*+ and Dbar^*0K^*0
    G1 = Gdr(EE, m1, m2, Λ)
    G2 = Gdr(EE, m3, m4, Λ)
    G = sp.diag([G1, G2], unpack=True)
    # Physical production amplitudes
    det = 1 - G2*V0p[1, 1] - G1*V0p[0, 0] - G1*G2*V0p[0, 1]**2 + G1*G2*V0p[0, 0]*V0p[1, 1]

    P = sp.Matrix([[UBp[0,0]-UBp[0,0]*G2*V0p[1,1]+UBp[0,1]*G2*V0p[1,0], 
                    UBp[0,1]-UBp[0,1]*G1*V0p[0,0]+UBp[0,0]*G1*V0p[0,1]]]) / det

    # Non-dynamic transitions
    Vpr0p = sp.Matrix([[E11+E10, E11-E10], 
                        [E11-E10, E11+E10]]) / 2
    
    Ppr = P * G * Vpr0p

    # Invariant mass distribution
    pKp = sp.sqrt(Kallen(EE**2, Kpm**2, Dpm**2) ) / 2 / EE
    pDp = sp.sqrt(Kallen(Bp**2, EE**2, Dpm**2) ) / 2 / Bp 

    dist = 1/2/sp.pi**3 * 1/16/Bp * sp.Abs(Ppr[ith, jth])**2 * sp.Abs(pKp) * sp.Abs(pDp)

    return dist

# B0 -> D0 D- K-
# 0+ channel
def channel0p_B0(EE, Λ, A, B, r0, C11, C10, E11, E10, ith, jth, m1=Dstpm, m2=Kstpm, m3=Dst0, m4=Kst0):
    UB0 = sp.Matrix([[B*r0, -A*r0]]) / sp.sqrt(2)

    V0p = sp.Matrix([[C11+C10, C11-C10], 
                    [C11-C10, C11+C10]]) / 2

    # propagators for D^*- K^*+ and Dbar^*0K^*0
    G1 = Gdr(EE, m1, m2, Λ)
    G2 = Gdr(EE, m3, m4, Λ)
    G = sp.diag([G1, G2], unpack=True)
    # Physical production amplitudes
    det = 1 - G2*V0p[1, 1] - G1*V0p[0, 0] - G1*G2*V0p[0, 1]**2 + G1*G2*V0p[0, 0]*V0p[1, 1]

    P = sp.Matrix([[UB0[0,0]-UB0[0,0]*G2*V0p[1,1]+UB0[0,1]*G2*V0p[1,0], 
                    UB0[0,1]-UB0[0,1]*G1*V0p[0,0]+UB0[0,0]*G1*V0p[0,1]]]) / det

    # Non-dynamic transitions
    Vpr0p = sp.Matrix([[E11+E10, E11-E10], 
                        [E11-E10, E11+E10]]) / 2
    
    Ppr = P * G * Vpr0p

    # Invariant mass distribution
    pKp = sp.sqrt(Kallen(EE**2, Kpm**2, Dpm**2) ) / 2 / EE
    pDp = sp.sqrt(Kallen(B0**2, EE**2, Dpm**2) ) / 2 / B0 

    dist = 1/2/sp.pi**3 * 1/16/B0 * sp.Abs(Ppr[ith, jth])**2 * sp.Abs(pKp) * sp.Abs(pDp)

    return dist
    
# B0 -> D*0 D*- K+
# 1+ channel
def channel1p(EE, Λ, A, B, r1, C11, C10, C30, C31, E11, E10, ith, jth, m1=Dstpm, m2=Kstpm, m3=Dst0, m4=Kst0, m5=Dpm, 
                m6 =D0):
    UB0 = sp.Matrix([[B, -A, B*r1, -A*r1]]) / sp.sqrt(2)

    V11 = (1/3*C11 + 2/3*C31 + 1/3*C10 + 2/3*C30) / 2
    V12 = (1/3*C11 + 2/3*C31 - 1/3*C10 - 2/3*C30) / 2
    V13 = (sp.sqrt(2)/3 * (C11-C31+C10-C30) ) / 2
    V14 = (sp.sqrt(2)/3 * (C11-C31-C10+C30) ) / 2

    V21 = V12
    V22 = (1/3*C11 + 2/3*C31 + 1/3*C10 + 2/3*C30) / 2
    V23 = (sp.sqrt(2)/3 * (C11-C31-C10+C30)) / 2
    V24 = (sp.sqrt(2)/3 * (C11-C31+C10-C30)) / 2

    V31 = V13
    V32 = V23
    V33 = (2/3*C11 + 1/3*C31 + 2/3*C10 + 1/3*C30) / 2
    V34 = (2/3*C11 + 1/3*C31 - 2/3*C10 - 1/3*C30) / 2

    V41 = V14
    V42 = V24
    V43 = V34
    V44 = (2/3*C11 + 1/3*C31 + 2/3*C10 + 1/3*C30) / 2

    #V1p = sp.Matrix([[V11, V12, V13, V14],
    #                [V21, V22, V23, V24],
    #                [V31, V32, V33, V34],
    #                [V41, V42, V43, V44]]) / 2
    
    # Two body propagator
    G1 = Gdr(EE, m5, m2, Λ)
    G2 = Gdr(EE, m6, m4, Λ)
    G3 = Gdr(EE, m1, m2, Λ)
    G4 = Gdr(EE, m3, m4, Λ)
    G = sp.diag([G1, G2, G3, G4], unpack=True)
    # Non-danamics transition
    Vpr1p = sp.Matrix([[E11+E10, E11-E10],
                      [E11-E10, E11+E10],
                      [E11+sp.sqrt(2)*E10, sp.sqrt(2)*(E11-E10)],
                      [sp.sqrt(2)*(E11-E10), sp.sqrt(2)*(E11+E10)]]) / 2 / sp.sqrt(3)
    # Physical production amp
    U1 = UB0[0, 0]; U2 = UB0[0, 1]; U3 = UB0[0, 2]; U4 = UB0[0, 3]

    F1 = G3*U3*V31 + G4*U4*V41 - G3*G4*U4*V33*V41 + G3*G4*U3*V34*V41 + G3*G4*U4*V31*V43 - G3*G4*U3*V31*V44 + G2*(G4*U4*(-(V22*V41) + V21*V42) + G3*G4*U4*(-(V23*V32*V41) + V22*V33*V41 + V23*V31*V42 - V21*V33*V42 - V22*V31*V43 + V21*V32*V43) + G3*U3*(G4*V24*(V32*V41 - V31*V42) - V22*(V31 + G4*V34*V41 - G4*V31*V44) + V21*(V32 + G4*V34*V42 - G4*V32*V44)) + U2*(V21 + G4*V24*V41 + G3*G4*V24*(-(V33*V41) + V31*V43) - G4*V21*V44 + G3*V23*(V31 + G4*V34*V41 - G4*V31*V44) - G3*V21*(V33 + G4*V34*V43 - G4*V33*V44))) - U1*(-1 + G3*V33 + G3*G4*V34*V43 + G4*V44 - G3*G4*V33*V44 + G2*(V22 + G4*V24*V42 + G3*G4*V24*(-(V33*V42) + V32*V43) - G4*V22*V44 + G3*V23*(V32 + G4*V34*V42 - G4*V32*V44) - G3*V22*(V33 + G4*V34*V43 - G4*V33*V44)))
    F2 = G3*U3*V32 + G4*U4*V42 - G3*G4*U4*V33*V42 + G3*G4*U3*V34*V42 + G3*G4*U4*V32*V43 - G3*G4*U3*V32*V44 - U2*(-1 + G3*V33 + G3*G4*V34*V43 + G4*V44 - G3*G4*V33*V44 + G1*(V11 + G4*V14*V41 + G3*G4*V14*(-(V33*V41) + V31*V43) - G4*V11*V44 + G3*V13*(V31 + G4*V34*V41 - G4*V31*V44) - G3*V11*(V33 + G4*V34*V43 - G4*V33*V44))) + G1*(G4*U4*(V12*V41 - V11*V42) + G3*G4*U4*(V13*V32*V41 - V12*V33*V41 - V13*V31*V42 + V11*V33*V42 + V12*V31*V43 - V11*V32*V43) + G3*U3*(G4*V14*(-(V32*V41) + V31*V42) + V12*(V31 + G4*V34*V41 - G4*V31*V44) - V11*(V32 + G4*V34*V42 - G4*V32*V44)) + U1*(V12 + G4*V14*V42 + G3*G4*V14*(-(V33*V42) + V32*V43) - G4*V12*V44 + G3*V13*(V32 + G4*V34*V42 - G4*V32*V44) - G3*V12*(V33 + G4*V34*V43 - G4*V33*V44)))
    F3 = G2*U2*V23 + G2*G4*U4*V23*V42 + G4*U4*V43 - G2*G4*U4*V22*V43 + G2*G4*U2*V24*V43 - G2*G4*U2*V23*V44 - U3*(-1 + G2*V22 + G2*G4*V24*V42 + G4*V44 - G2*G4*V22*V44 + G1*(V11 + G4*V14*V41 + G2*G4*V14*(-(V22*V41) + V21*V42) - G4*V11*V44 + G2*V12*(V21 + G4*V24*V41 - G4*V21*V44) - G2*V11*(V22 + G4*V24*V42 - G4*V22*V44))) + G1*(G4*U4*(V13*V41 - V11*V43) + G2*G4*U4*(-(V13*V22*V41) + V12*V23*V41 + V13*V21*V42 - V11*V23*V42 - V12*V21*V43 + V11*V22*V43) + G2*U2*(G4*V14*(-(V23*V41) + V21*V43) + V13*(V21 + G4*V24*V41 - G4*V21*V44) - V11*(V23 + G4*V24*V43 - G4*V23*V44)) + U1*(V13 + G4*V14*V43 + G2*G4*V14*(V23*V42 - V22*V43) - G4*V13*V44 - G2*V13*(V22 + G4*V24*V42 - G4*V22*V44) + G2*V12*(V23 + G4*V24*V43 - G4*V23*V44)))
    F4 = G2*U2*V24 + G2*G3*U3*V24*V32 - G2*G3*U2*V24*V33 - U4*(-1 + G2*V22 + G2*G3*V23*V32 + G3*V33 - G2*G3*V22*V33 + G1*(V11 + G3*V13*V31 + G2*G3*V13*(-(V22*V31) + V21*V32) - G3*V11*V33 + G2*V12*(V21 + G3*V23*V31 - G3*V21*V33) - G2*V11*(V22 + G3*V23*V32 - G3*V22*V33))) + G3*U3*V34 - G2*G3*U3*V22*V34 + G2*G3*U2*V23*V34 + G1*(G3*U3*(V14*V31 - V11*V34) + G2*G3*U3*(-(V14*V22*V31) + V12*V24*V31 + V14*V21*V32 - V11*V24*V32 - V12*V21*V34 + V11*V22*V34) + G2*U2*(V14*(V21 + G3*V23*V31 - G3*V21*V33) + G3*V13*(-(V24*V31) + V21*V34) + V11*(-V24 + G3*V24*V33 - G3*V23*V34)) + U1*(V14 - G3*V14*V33 - G2*V14*(V22 + G3*V23*V32 - G3*V22*V33) + G3*V13*V34 + G2*G3*V13*(V24*V32 - V22*V34) + G2*V12*(V24 - G3*V24*V33 + G3*V23*V34)))
    det = G4*(-(G1*V14) + G1*G2*V14*V22 - G1*G2*V12*V24 + G1*G2*G3*V14*V23*V32 - G1*G2*G3*V13*V24*V32 + G1*G3*V14*V33 - G1*G2*G3*V14*V22*V33 + G1*G2*G3*V12*V24*V33 - G1*G3*V13*V34 + G1*G2*G3*V13*V22*V34 - G1*G2*G3*V12*V23*V34)*V41 - G4*(G1*G2*V14*V21 + G2*V24 - G1*G2*V11*V24 + G1*G2*G3*V14*V23*V31 - G1*G2*G3*V13*V24*V31 - G1*G2*G3*V14*V21*V33 - G2*G3*V24*V33 + G1*G2*G3*V11*V24*V33 + G1*G2*G3*V13*V21*V34 + G2*G3*V23*V34 - G1*G2*G3*V11*V23*V34)*V42 + G4*(-(G1*G3*V14*V31) + G1*G2*G3*V14*V22*V31 - G1*G2*G3*V12*V24*V31 - G1*G2*G3*V14*V21*V32 - G2*G3*V24*V32 + G1*G2*G3*V11*V24*V32 - G3*V34 + G1*G3*V11*V34 + G1*G2*G3*V12*V21*V34 + G2*G3*V22*V34 - G1*G2*G3*V11*V22*V34)*V43 + (1 - G1*V11 - G1*G2*V12*V21 - G2*V22 + G1*G2*V11*V22 - G1*G3*V13*V31 + G1*G2*G3*V13*V22*V31 - G1*G2*G3*V12*V23*V31 - G1*G2*G3*V13*V21*V32 - G2*G3*V23*V32 + G1*G2*G3*V11*V23*V32 - G3*V33 + G1*G3*V11*V33 + G1*G2*G3*V12*V21*V33 + G2*G3*V22*V33 - G1*G2*G3*V11*V22*V33)*(1 - G4*V44)
    
    P = sp.Matrix([[F1, F2, F3, F4]]) / det

    Ppr = P * G * Vpr1p

    pKp = sp.sqrt(Kallen(EE**2, Kpm**2, Dstpm**2) ) / 2 / EE
    pD0 = sp.sqrt(Kallen(B0**2, EE**2, Dst0) ) / 2 / B0
    dist = 1/2/sp.pi**3 * 1/16/B0 * sp.Abs(Ppr[ith, jth])**2 * sp.Abs(pKp) * sp.Abs(pD0)

    return dist