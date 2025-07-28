import numpy as np


def multibody_response(B_c, B_f, B_s, K_f, K_s, m_c, m_f, m_s, w,
                       K_p, B_p, F_f_mag, F_f_phase, F_s_mag, F_s_phase):
    # Define all intermediate terms (translated from MATLAB)
    t2 = np.abs(w)
    t3 = B_c * w
    t4 = B_f * w
    t5 = B_s * w
    t6 = K_f * w
    t7 = B_c ** 2
    t8 = m_c ** 2
    t9 = w ** 2
    t10 = w ** 3
    t12 = K_f * K_s
    t16 = -K_f
    t17 = -K_s
    t18 = 1.0 / w
    t25 = F_f_phase * 1j
    t26 = F_s_phase * 1j
    t27 = K_f * 1j
    t28 = K_s * 1j
    t34 = m_c * w * 1j
    t11 = t9 ** 2
    t13 = 2.0 * t3
    t14 = K_s * t4
    t15 = K_f * t5
    t19 = -t6
    t20 = 1.0 / t2
    t21 = m_c * t10
    t22 = m_f * t9
    t23 = m_f * t10
    t24 = m_s * t9
    t29 = -t12
    t30 = np.exp(t25)
    t31 = np.exp(t26)
    t33 = t5 * 1j
    t37 = 2.0 * m_c * t9
    t38 = -t25
    t39 = -t26
    t40 = -t27
    t41 = -t28
    t42 = t4 * t5
    t45 = m_s * t6 * w
    t47 = t12 * 1j
    t56 = t3 ** 2
    t62 = t3 * w * 1j
    t63 = t4 * w * 1j
    t64 = m_c * t9 * 1j
    t65 = 2.0j * m_c * t9
    t68 = B_c + t34
    t69 = -2.0 * m_c * t3 * t9
    t73 = K_p * t18 * 1j
    t89 = -2.0j * m_c * t3 * t9
    t32 = m_f * m_s * t11
    t35 = -t14
    t36 = -t15
    t43 = t4 * t24
    t44 = t5 * t22
    t46 = K_s * t22
    t48 = t8 * t11
    t49 = 1.0 / t30
    t50 = 1.0 / t31
    t51 = t14 * 1j
    t52 = t15 * 1j
    t53 = -t21
    t54 = -t23
    t55 = -t24
    t57 = m_c * t9 * t13
    t58 = -t47
    t66 = t22 * 1j
    t67 = t24 * 1j
    t74 = -t64
    t75 = -t65
    t78 = t4 * t33
    t79 = -t56
    t80 = t3 * t65
    t82 = t22 * t33
    t83 = t45 * 1j
    t84 = t22 * t28
    t87 = -t73
    t88 = t56 * 1j
    t92 = B_p + t73
    t93 = t4 + t5 + t13
    t95 = t3 + t64
    t100 = t16 + t17 + t22 + t24 + t37
    t105 = t19 + t21 + t23 + t62 + t63
    t59 = -t32
    t60 = -t51
    t61 = -t52
    t70 = -t43
    t71 = -t44
    t72 = t32 * 1j
    t76 = -t66
    t77 = -t67
    t81 = t43 * 1j
    t85 = t48 * 1j
    t91 = -1j * t44
    t94 = -t88
    t96 = B_p + t87
    t97 = t93 ** 2
    t98 = K_s + t33 + t55
    t99 = t4 + t40 + t66
    t101 = t100 ** 2
    t103 = t5 + t41 + t67 + t95
    t106 = t6 + t53 + t54 + t62 + t63
    t109 = t40 + t41 + t65 + t66 + t67 + t93
    t86 = -t72
    t90 = -t81
    t102 = t95 + t99
    t104 = t3 + t5 + t28 + t74 + t77
    t107 = t97 + t101
    t110 = t27 + t28 + t75 + t76 + t77 + t93
    t111 = 1.0 / t109
    t116 = t29 + t42 + t45 + t46 + t48 + t59 + t60 + t61 + t79 + t81 + t82 + t89
    t125 = -1.0 / (t14 + t15 - t42 * 1j - t46 * 1j + t47 + t57 + t70 + t71 + t72 - t83 - t85 + t88)
    t108 = np.sqrt(t107)
    t112 = 1.0 / t110
    t113 = F_f_mag * t30 * t103 * t111
    t115 = t29 + t42 + t45 + t46 + t48 + t51 + t52 + t59 + t79 + t80 + t90 + t91
    t117 = np.abs(t116)
    t119 = -1.0 / (t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w)
    t120 = 1.0 / (t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) ** 2
    t122 = t14 + t15 + t57 + t58 + t70 + t71 + t78 + t83 + t84 + t85 + t86 + t94
    t123 = t35 + t36 + t43 + t44 + t58 + t69 + t78 + t83 + t84 + t85 + t86 + t94
    t138 = -1.0 / ((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0)
    t114 = F_f_mag * t49 * t104 * t112
    t118 = 1.0 / t115
    t121 = 1.0 / t117
    t124 = 1.0 / t122
    t133 = t96 * t109 * t119 * w
    t140 = F_s_mag * t18 * t50 * t105 * t112 * t115 * t125
    t142 = (F_s_mag * t18 * t50 * t105 * t112 * t115) / (
                t14 + t15 - t42 * 1j - t46 * 1j + t47 + t57 + t70 + t71 + t72 - t83 - t85 + t88)
    t126 = F_s_mag * t9 * t31 * t68 * t124 * 1j
    t127 = F_f_mag * t30 * t98 * t124 * w
    t128 = F_f_mag * t30 * t95 * t124 * w * 1j
    t131 = F_s_mag * t31 * t99 * t124 * w * 1j
    t132 = t92 * t110 * t118 * w
    t135 = t133 + 1.0
    t141 = -F_s_mag * t18 * t31 * t106 * t111 * t124 * (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w)
    t143 = -t141
    t145 = t114 + t142
    t129 = -t128
    t130 = -t127
    t134 = t132 + 1.0
    t136 = np.abs(t135)
    t144 = t113 + t143
    t137 = 1.0 / t134
    t139 = 1.0 / t136

    mag_U = t108 * t121 * t139 * np.abs(t96 * t144 * w)
    phase_U = np.angle((t133 * t144) / ((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0))

    t146 = (t9 * t96 * t103 * t109 * t119 * t144) / (((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0) * (
                                                                 t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w))
    t147 = (t9 * t96 * t102 * t109 * t124 * t144 * 1j) / (((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0) * (
                                                                      t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w))
    t148 = t126 + t130 + t146
    t149 = t129 + t131 + t147
    t150 = (t9 * t96 * t109 * t110 * t118 * t137 * t144 * t145) / (((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0) * (
                                                                               t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w))

    real_P = np.real(t150) / 2.0
    imag_P = np.imag(t150) / 2.0
    mag_X_u = t108 * t121 * t139 * np.abs(t144)
    phase_X_u = np.angle((t109 * t144 * 1j) / (((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0) * (
                                                           t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w)))
    mag_X_f = t20 * np.abs(t148)
    phase_X_f = np.angle(t18 * (-t126 + t127 + (t9 * t96 * t103 * t109 * t120 * t144) / ((t96 * t109 * w) / (
                t12 + t32 - t42 - t48 + t51 + t52 + t56 + t80 + t90 + t91 + t17 * t22 + m_s * t19 * w) - 1.0)) * 1j)
    mag_X_s = t20 * np.abs(t149)
    phase_X_s = np.angle(t18 * t149 * 1j)

    return mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s
