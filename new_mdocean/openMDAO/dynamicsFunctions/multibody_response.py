import numpy as np

def  multibody_response(B_c,B_f,B_s,K_f,K_s,m_c,m_f,m_s,w,K_p,B_p,F_f_mag,F_f_phase,F_s_mag,F_s_phase):
    t2 = np.cos(F_f_phase)
    t3 = np.cos(F_s_phase)
    t4 = np.sin(F_f_phase)
    t5 = np.sin(F_s_phase)
    t6 = B_c * w
    t7 = B_f * w
    t8 = B_p * w
    t9 = B_s * w
    t10 = K_f * w

    t11 = B_c ** 2  # B_c squared
    t12 = B_c ** 3  # B_c cubed
    t14 = B_f ** 2  # B_f squared
    t15 = B_p ** 2  # B_p squared
    t16 = B_s ** 2  # B_s squared
    t17 = K_f ** 2  # K_f squared
    t18 = K_p ** 2  # K_p squared
    t19 = K_s ** 2  # K_s squared
    t20 = m_c ** 2  # m_c squared
    t21 = m_c ** 3  # m_c cubed
    t23 = m_f ** 2  # m_f squared
    t24 = m_s ** 2  # m_s squared
    t25 = w ** 2  # w squared
    t26 = w ** 3  # w cubed
    t28 = w ** 5  # w to the 5th power
    t30 = w ** 7  # w to the 7th power

    t31 = B_f * K_s  # element-wise product
    t32 = B_s * K_f  # element-wise product
    t33 = K_f * K_s  # element-wise product

    t45 = -K_f  # negative K_f
    t46 = -K_s  # negative K_s

    t47 = 1.0 / w  # element-wise reciprocal of w

    t60 = K_f * 1j  # complex K_f * i
    t61 = K_p * 1j  # complex K_p * i
    t62 = K_s * 1j  # complex K_s * i

    t13 = t11 ** 2  # (B_c^2)^2 = B_c^4
    t22 = t20 ** 2  # (m_c^2)^2 = m_c^4
    t27 = t25 ** 2  # (w^2)^2 = w^4
    t29 = t25 ** 3  # (w^2)^3 = w^6

    t34 = t6 * 2.0  # double of t6
    t35 = B_c * t2  # element-wise product

    t36 = B_c * t3  # B_c multiplied by t3
    t37 = K_f * t3  # K_f multiplied by t3
    t38 = K_s * t2  # K_s multiplied by t2
    t39 = B_c * t4  # B_c multiplied by t4
    t40 = B_c * t5  # B_c multiplied by t5
    t41 = K_s * t7  # K_s multiplied by t7
    t42 = K_f * t9  # K_f multiplied by t9
    t43 = K_f * t5  # K_f multiplied by t5
    t44 = K_s * t4  # K_s multiplied by t4

    t49 = -t8  # Negative t8
    t50 = -t10  # Negative t10

    t51 = t6 * w  # B_c * w * w
    t52 = t7 * w  # B_f * w * w

    t55 = m_c * t25  # m_c * w^2
    t56 = m_c * t26  # m_c * w^3
    t57 = m_f * t25  # m_f * w^2
    t58 = m_f * t26  # m_f * w^3
    t59 = m_s * t25  # m_s * w^2

    t65 = t3 * t7  # cos(F_s_phase) * B_f * w
    t66 = t2 * t9  # cos(F_f_phase) * B_s * w
    t67 = t5 * t7  # sin(F_s_phase) * B_f * w
    t68 = t4 * t9  # sin(F_f_phase) * B_s * w

    t69 = m_c * t2 * w  # m_c * cos(F_f_phase) * w
    t70 = m_c * t3 * w  # m_c * cos(F_s_phase) * w
    t71 = -t33  # negative of K_f * K_s
    t72 = m_c * t4 * w  # m_c * sin(F_f_phase) * w
    t73 = m_c * t5 * w  # m_c * sin(F_s_phase) * w

    t74 = F_f_mag * t2 * t6  # F_f_mag * cos(F_f_phase) * B_c * w
    t76 = F_s_mag * t3 * t6  # F_s_mag * cos(F_s_phase) * B_c * w
    t79 = F_f_mag * t4 * t6  # F_f_mag * sin(F_f_phase) * B_c * w
    t81 = F_s_mag * t5 * t6  # F_s_mag * sin(F_s_phase) * B_c * w

    t83 = t4 * 1j  # sin(F_f_phase) * i
    t84 = t5 * 1j  # sin(F_s_phase) * i

    t88 = -t60  # negative of K_f * i
    t89 = -t62  # negative of K_s * i

    t90 = t7 * t9  # B_f * w * B_s * w

    t92 = m_f * t9 * w  # m_f * B_s * w^2
    t95 = m_s * t10 * w  # m_s * K_f * w^2

    t97 = t33 * 1j  # K_f * K_s * i

    t99 = F_f_mag * K_p * t2 * t33  # F_f_mag * K_p * cos(F_f_phase) * K_f * K_s
    t100 = F_s_mag * K_p * t3 * t33  # F_s_mag * K_p * cos(F_s_phase) * K_f * K_s

    t101 = F_f_mag * K_p * t4 * t33
    t102 = F_s_mag * K_p * t5 * t33
    t109 = t6 + t9
    t111 = K_f * K_p * t19 * 2.0
    t112 = t18 * t33 * 2.0
    t113 = K_p * K_s * t17 * 2.0
    t115 = t17 * t18
    t116 = t17 * t19
    t117 = t18 * t19
    t118 = t6 ** 2
    t131 = F_f_mag * K_f * t2 * t19
    t132 = F_f_mag * K_p * t2 * t19
    t133 = F_s_mag * K_p * t3 * t17
    t134 = F_s_mag * K_s * t3 * t17
    t135 = F_f_mag * K_f * t4 * t19
    t136 = F_f_mag * K_p * t4 * t19
    t137 = F_s_mag * K_p * t5 * t17
    t138 = F_s_mag * K_s * t5 * t17
    t149 = F_f_mag * t2 * t8 * t33
    t155 = F_s_mag * t3 * t8 * t33
    t169 = F_f_mag * t4 * t8 * t33
    t175 = F_s_mag * t5 * t8 * t33
    t182 = t6 ** 3 * t8 * 4.0
    t190 = t6 * t8 * t33 * 4.0
    t206 = m_c * t8 * t9 * t10 * w * 4.0
    t209 = K_p * K_s * m_c * t10 * w * 4.0
    t210 = K_p * K_s * m_f * t10 * w * 4.0
    t213 = F_f_mag * t2 * t18 * t33
    t214 = F_s_mag * t3 * t18 * t33
    t228 = F_f_mag * t2 * t7 * t19
    t229 = F_f_mag * t2 * t8 * t19
    t230 = F_s_mag * t3 * t8 * t17
    t231 = F_s_mag * t3 * t9 * t17
    t233 = F_f_mag * t4 * t7 * t19
    t234 = F_f_mag * t4 * t8 * t19
    t235 = F_s_mag * t5 * t8 * t17
    t236 = F_s_mag * t5 * t9 * t17
    t394 = F_s_mag * t3 * t17 * t46
    t415 = F_s_mag * t5 * t17 * t46
    t425 = t6 * t7 * t8 ** 2 * 4.0
    t428 = t6 * t8 ** 2 * t9 * 4.0
    t432 = t6 * t7 * t18 * 4.0
    t433 = t6 * t9 * t18 * 4.0
    t434 = t7 * t8 * t19 * 2.0
    t436 = t8 * t9 * t17 * 2.0
    t443 = t8 ** 2 * t33 * 2.0
    t451 = t7 ** 2 * t8 ** 2
    t453 = t8 ** 2 * t9 ** 2
    t456 = t7 ** 2 * t18
    t457 = t8 ** 2 * t17
    t458 = t7 ** 2 * t19
    t459 = t9 ** 2 * t17
    t460 = t8 ** 2 * t19
    t461 = t9 ** 2 * t18
    t473 = m_c * t8 ** 2 * t10 * w * 4.0
    t476 = m_f * t8 ** 2 * t10 * w * 2.0
    t493 = K_p * t10 * t20 * t26 * 2.0
    t494 = m_c * t10 * t18 * w * 4.0
    t495 = m_f * t10 * t18 * w * 2.0
    t496 = K_s * t10 * t20 * t26 * 2.0
    t497 = m_f * t10 * t19 * w * 2.0
    t502 = K_p * m_s * t10 ** 2 * 2.0
    t504 = K_p * t10 * t24 * t26 * 2.0
    t506 = K_s * m_s * t10 ** 2 * 2.0
    t525 = t10 ** 2 * t24 * t25
    t540 = m_s * t10 * t20 * t28 * 2.0
    t542 = m_f * t10 * t24 * t28 * 2.0
    t616 = F_s_mag * m_f * t3 * t8 * t10 * w * 2.0
    t633 = F_s_mag * K_p * m_f * t3 * t10 * w * 2.0
    t634 = F_s_mag * K_s * m_f * t3 * t10 * w * 2.0
    t643 = F_s_mag * m_f * t5 * t8 * t10 * w * 2.0
    t663 = F_s_mag * K_p * m_f * t5 * t10 * w * 2.0
    t664 = F_s_mag * K_s * m_f * t5 * t10 * w * 2.0
    t668 = F_f_mag * t2 * t10 * t24 * t26
    t670 = F_s_mag * t3 * t10 * t20 * t26
    t673 = F_s_mag * m_s * t3 * t10 ** 2
    t713 = F_f_mag * t4 * t10 * t24 * t26
    t715 = F_s_mag * t5 * t10 * t20 * t26
    t717 = F_s_mag * m_s * t5 * t10 ** 2
    t1243 = F_s_mag * K_s * m_f * t5 * t8 * t10 * w * -2.0
    t48 = t27 ** 2
    t53 = F_f_mag * t38
    t54 = F_s_mag * t37
    t63 = F_f_mag * t44
    t64 = F_s_mag * t43
    t75 = F_f_mag * t66
    t77 = F_s_mag * t65
    t78 = m_f * m_s * t27
    t80 = F_f_mag * t68
    t82 = F_s_mag * t67
    t85 = -t37
    t86 = -t38
    t87 = t55 * 2.0
    t91 = m_s * t52
    t93 = t7 * t59
    t94 = t9 * t57
    t96 = K_s * t57
    t98 = t20 * t27
    t103 = t3 * t57
    t104 = t2 * t59
    t105 = t5 * t57
    t106 = t4 * t59
    t107 = t41 * 1j
    t108 = t42 * 1j
    t119 = t118 ** 2
    t120 = m_c * t34 * w
    t121 = t34 * t55
    t122 = -t97
    t123 = -t72
    t124 = -t73
    t126 = -t76

    t129 = -t79
    t141 = F_f_mag * t2 * t55
    t143 = F_s_mag * t3 * t55
    t145 = K_f * K_p * t74
    t146 = t33 * t74
    t150 = F_f_mag * K_p * t2 * t42
    t152 = t33 * t76
    t153 = K_p * K_s * t76
    t154 = F_s_mag * K_p * t3 * t41
    t157 = t55 * 1j
    t158 = t55 * 2.0j
    t159 = t57 * 1j
    t160 = t59 * 1j
    t161 = F_f_mag * t4 * t55
    t163 = F_s_mag * t5 * t55
    t165 = K_f * K_p * t79
    t166 = t33 * t79
    t168 = F_f_mag * K_p * t4 * t41
    t170 = F_f_mag * K_p * t4 * t42
    t172 = t33 * t81
    t173 = K_p * K_s * t81
    t174 = F_s_mag * K_p * t5 * t41
    t176 = F_s_mag * K_p * t5 * t42
    t178 = -t92
    t183 = t2 + t83
    t184 = t3 + t84
    t185 = F_s_mag * K_p * t3 * t71
    t186 = t6 * t8 * t90 * 4.0
    t187 = F_s_mag * K_p * t5 * t71
    t188 = K_p * t21 * t29 * 4.0
    t189 = K_p * t6 * t41 * 4.0
    t191 = K_p * t6 * t42 * 4.0
    t194 = K_p * t6 * t7 * t55 * 4.0
    t195 = m_c * t8 * t10 * t51 * 4.0
    t196 = t6 * t41 * t55 * 4.0
    t197 = m_c * t9 * t10 * t51 * 4.0
    t198 = K_s * t6 * t8 * t55 * 4.0
    t199 = K_p * t6 * t9 * t55 * 4.0
    t201 = m_s * t8 * t10 * t51 * 4.0
    t204 = t8 * t41 * t55 * 4.0
    t205 = K_p * t55 * t90 * 4.0
    t207 = t8 * t10 * t92 * 4.0
    t208 = t8 * t41 * t59 * 4.0
    t211 = K_p * K_s * t95 * 4.0
    t212 = K_p * t131
    t215 = K_s * t133
    t216 = m_c * m_f * t6 * t8 * t27 * 4.0
    t217 = m_c * m_s * t6 * t7 * t27 * 4.0
    t218 = m_c * m_f * t6 * t9 * t27 * 4.0
    t219 = m_c * m_s * t6 * t8 * t27 * 4.0
    t221 = m_c * m_s * t7 * t8 * t27 * 4.0
    t222 = m_c * m_f * t8 * t9 * t27 * 4.0
    t223 = K_p * m_s * t10 * t56 * 4.0
    t224 = K_p * K_s * m_c * m_f * t27 * 4.0
    t225 = K_p * m_s * t10 * t58 * 4.0
    t226 = K_s * m_s * t10 * t58 * 4.0
    t232 = t90 * 1j
    t237 = K_p * m_c * m_f * m_s * t29 * 4.0
    t238 = -t118
    t242 = t95 * 1j
    t243 = t57 * t62
    t247 = F_f_mag * t4 * t21 * t29
    t248 = F_s_mag * t5 * t21 * t29
    t249 = K_p * t7 * t79
    t250 = K_f * t8 * t79
    t251 = t41 * t79
    t252 = t42 * t79
    t254 = F_f_mag * t4 * t8 * t41
    t256 = F_f_mag * t4 * t8 * t42
    t258 = t41 * t81
    t259 = t42 * t81
    t260 = K_s * t8 * t81
    t261 = K_p * t9 * t81
    t262 = F_s_mag * t5 * t8 * t41
    t264 = F_s_mag * t5 * t8 * t42
    t265 = K_f * t18 * t79
    t267 = t7 * t136
    t268 = F_f_mag * t4 * t18 * t41
    t269 = t8 * t135
    t270 = F_f_mag * t4 * t18 * t42
    t272 = K_s * t18 * t81
    t273 = F_s_mag * t5 * t18 * t41
    t274 = t8 * t138
    t275 = F_s_mag * t5 * t18 * t42
    t276 = t9 * t137
    t278 = t8 * t57 * t74
    t283 = t8 * t59 * t74
    t287 = t8 * t57 * t76
    t292 = t8 * t59 * t76
    t296 = -t182
    t297 = K_p * t57 * t74
    t299 = F_f_mag * t8 * t10 * t69
    t300 = F_f_mag * m_s * t2 * t10 * t51
    t304 = K_p * t59 * t74
    t307 = F_f_mag * t2 * t8 * t95
    t310 = K_p * t57 * t76
    t312 = F_s_mag * m_s * t3 * t10 * t51
    t315 = F_s_mag * t9 * t10 * t70
    t316 = K_p * t59 * t76
    t320 = F_s_mag * t3 * t8 * t95
    t323 = t8 * t57 * t79

    t328 = t8 * t59 * t79
    t332 = t8 * t57 * t81
    t337 = t8 * t59 * t81
    t341 = F_f_mag * K_p * t10 * t69
    t343 = F_f_mag * K_p * t2 * t95
    t345 = F_s_mag * K_s * t10 * t70
    t347 = F_s_mag * K_p * t3 * t95
    t349 = K_p * t57 * t79
    t351 = F_f_mag * t8 * t10 * t72
    t352 = F_f_mag * m_s * t4 * t10 * t51
    t356 = K_p * t59 * t79
    t359 = F_f_mag * t4 * t8 * t95
    t362 = K_p * t57 * t81
    t364 = F_s_mag * m_s * t5 * t10 * t51
    t367 = F_s_mag * t9 * t10 * t73
    t368 = K_p * t59 * t81
    t372 = F_s_mag * t5 * t8 * t95
    t375 = F_f_mag * K_p * t10 * t72
    t377 = F_f_mag * K_p * t4 * t95
    t379 = F_s_mag * K_s * t10 * t73
    t381 = F_s_mag * K_p * t5 * t95
    t383 = F_f_mag * m_c * m_f * t2 * t8 * t27
    t385 = F_f_mag * m_c * m_s * t2 * t7 * t27
    t390 = F_s_mag * m_c * m_f * t3 * t9 * t27
    t391 = F_s_mag * m_c * m_s * t3 * t8 * t27
    t393 = -t133
    t395 = F_f_mag * K_p * m_c * m_f * t2 * t27
    t396 = F_f_mag * m_s * t2 * t10 * t56
    t399 = F_s_mag * m_s * t3 * t10 * t56
    t400 = F_s_mag * K_s * m_c * m_f * t3 * t27
    t401 = F_s_mag * K_p * m_c * m_s * t3 * t27
    t403 = -t190
    t404 = F_f_mag * m_c * m_f * t4 * t8 * t27
    t406 = F_f_mag * m_c * m_s * t4 * t7 * t27
    t411 = F_s_mag * m_c * m_f * t5 * t9 * t27
    t412 = F_s_mag * m_c * m_s * t5 * t8 * t27
    t414 = -t137
    t416 = F_f_mag * K_p * m_c * m_f * t4 * t27
    t417 = F_f_mag * m_s * t4 * t10 * t56
    t420 = F_s_mag * m_s * t5 * t10 * t56
    t421 = F_s_mag * K_s * m_c * m_f * t5 * t27
    t422 = F_s_mag * K_p * m_c * m_s * t5 * t27
    t424 = t6 * t7 * t8 * t34
    t426 = t6 * t34 * t90
    t427 = t6 * t8 * t9 * t34
    t429 = t8 * t9 * t90 * 2.0
    t430 = t8 ** 2 * t90 * 2.0
    t431 = t7 * t8 * t90 * 2.0
    t435 = t18 * t90 * 2.0
    t437 = F_f_mag * m_c * m_f * m_s * t2 * t29
    t438 = F_s_mag * m_c * m_f * m_s * t3 * t29
    t439 = K_f * K_p * t6 * t34
    t440 = t6 * t33 * t34
    t441 = K_p * K_s * t6 * t34
    t442 = K_p * t7 * t41 * 2.0
    t444 = K_p * t9 * t42 * 2.0
    t448 = -t206
    t452 = t90 ** 2
    t454 = F_f_mag * m_c * m_f * m_s * t4 * t29
    t455 = F_s_mag * m_c * m_f * m_s * t5 * t29
    t462 = -t209
    t463 = -t210
    t469 = t7 * t8 * t24 * t27 * 2.0
    t470 = t8 * t9 * t23 * t27 * 2.0
    t471 = K_p * t55 * t118 * 4.0
    t472 = K_p * t6 * t34 * t57
    t474 = m_s * t10 * t34 * t51
    t477 = t9 * t10 * t92 * 2.0
    t478 = K_p * t6 * t34 * t59
    t479 = K_s * t8 ** 2 * t55 * 4.0
    t481 = t8 ** 2 * t95 * 2.0
    t485 = K_s * t8 ** 2 * t59 * 2.0
    t499 = K_s * t18 * t55 * 4.0
    t500 = t18 * t95 * 2.0
    t501 = K_p * t19 * t57 * 2.0
    t505 = K_p * K_s * t23 * t27 * 2.0
    t507 = K_s * t18 * t59 * 2.0
    t510 = F_f_mag * t2 * t33 * t49
    t511 = t71 * t76
    t512 = K_p * t46 * t76
    t516 = t8 ** 2 * t23 * t27
    t517 = t7 ** 2 * t24 * t27
    t518 = t9 ** 2 * t23 * t27
    t519 = t8 ** 2 * t24 * t27
    t520 = F_f_mag * t2 * t117
    t521 = F_s_mag * t3 * t115
    t522 = t39 + t69
    t523 = t40 + t70
    t524 = t18 * t23 * t27
    t526 = t19 * t23 * t27
    t527 = t18 * t24 * t27
    t528 = K_p * t45 * t79
    t529 = t71 * t79
    t532 = F_s_mag * t5 * t33 * t49
    t534 = m_c * m_f * t8 ** 2 * t27 * 4.0
    t536 = m_c * m_s * t8 ** 2 * t27 * 4.0
    t538 = K_p * m_f * t20 * t29 * 2.0
    t539 = m_c * m_f * t18 * t27 * 4.0
    t541 = K_s * m_f * t20 * t29 * 2.0
    t543 = K_p * m_s * t20 * t29 * 2.0

    t544 = m_c * m_s * t18 * t27 * 4.0
    t546 = K_p * m_f * t24 * t29 * 2.0
    t547 = K_p * m_s * t23 * t29 * 2.0
    t548 = K_s * m_s * t23 * t29 * 2.0
    t549 = t74 * t118
    t550 = t76 * t118
    t552 = t118 * 1j
    t553 = t79 * t118
    t554 = t81 * t118
    t555 = t6 * t55 * -2.0j
    t556 = t7 * t8 * t74
    t560 = t8 * t9 * t76
    t562 = F_f_mag * t2 * t21 * t29
    t563 = F_s_mag * t3 * t21 * t29
    t564 = K_p * t7 * t74
    t565 = K_f * t8 * t74
    t567 = t42 * t74
    t571 = F_f_mag * t2 * t8 * t42
    t573 = t41 * t76
    t575 = K_s * t8 * t76
    t576 = K_p * t9 * t76
    t577 = F_s_mag * t3 * t8 * t41
    t580 = t7 * t8 * t79
    t584 = t8 * t9 * t81
    t611 = F_s_mag * t10 * t34 * t70
    t614 = F_s_mag * t8 * t10 * t70 * 3.0
    t617 = F_s_mag * t3 * t10 * t92 * 2.0
    t618 = t49 + t61
    t631 = F_s_mag * K_p * t10 * t70 * 3.0
    t638 = F_s_mag * t10 * t34 * t73
    t641 = F_s_mag * t8 * t10 * t73 * 3.0
    t644 = F_s_mag * t5 * t10 * t92 * 2.0
    t645 = t6 * t55 * t74
    t647 = t6 * t59 * t74
    t650 = F_f_mag * t2 * t7 * t24 * t27
    t651 = t6 * t55 * t76
    t653 = t6 * t57 * t76
    t655 = F_f_mag * t2 * t8 * t24 * t27
    t656 = F_s_mag * t3 * t8 * t23 * t27
    t658 = F_s_mag * t3 * t9 * t23 * t27
    t661 = F_s_mag * K_p * t10 * t73 * 3.0
    t667 = F_f_mag * t2 * t19 * t57
    t669 = F_f_mag * K_p * t2 * t24 * t27
    t672 = F_s_mag * K_p * t3 * t23 * t27
    t674 = F_s_mag * K_s * t3 * t23 * t27
    t681 = F_f_mag * m_c * m_s * t2 * t27 * t34
    t682 = F_s_mag * m_c * m_f * t3 * t27 * t34
    t683 = F_f_mag * m_c * m_s * t2 * t8 * t27 * 3.0
    t684 = F_s_mag * m_c * m_f * t3 * t8 * t27 * 3.0
    t685 = t6 * t55 * t79
    t688 = t6 * t59 * t79
    t691 = F_f_mag * t4 * t7 * t24 * t27
    t692 = t6 * t55 * t81
    t694 = t6 * t57 * t81
    t696 = F_f_mag * t4 * t8 * t24 * t27
    t698 = F_s_mag * t5 * t8 * t23 * t27
    t700 = F_s_mag * t5 * t9 * t23 * t27
    t707 = F_f_mag * K_p * m_c * m_s * t2 * t27 * 3.0
    t709 = F_s_mag * K_p * m_c * m_f * t3 * t27 * 3.0
    t710 = F_s_mag * m_s * t3 * t10 * t58 * 2.0
    t712 = F_f_mag * t4 * t19 * t57
    t714 = F_f_mag * K_p * t4 * t24 * t27
    t716 = F_s_mag * K_p * t5 * t23 * t27
    t718 = F_s_mag * K_s * t5 * t23 * t27
    t719 = t51 + t52
    t720 = F_f_mag * m_c * m_s * t4 * t27 * t34
    t721 = F_s_mag * m_c * m_f * t5 * t27 * t34
    t722 = F_f_mag * m_c * m_s * t4 * t8 * t27 * 3.0
    t723 = F_s_mag * m_c * m_f * t5 * t8 * t27 * 3.0
    t724 = F_f_mag * K_p * m_c * m_s * t4 * t27 * 3.0
    t726 = F_s_mag * K_p * m_c * m_f * t5 * t27 * 3.0
    t727 = F_s_mag * m_s * t5 * t10 * t58 * 2.0
    t728 = t7 * t8 * t118 * -2.0
    t729 = t90 * t118 * -2.0
    t730 = t8 * t9 * t118 * -2.0
    t731 = F_f_mag * m_s * t2 * t20 * t29
    t732 = F_f_mag * m_f * t2 * t24 * t29
    t733 = F_s_mag * m_f * t3 * t20 * t29
    t734 = F_s_mag * m_s * t3 * t23 * t29
    t755 = F_f_mag * m_s * t4 * t20 * t29
    t756 = F_f_mag * m_f * t4 * t24 * t29
    t757 = F_s_mag * m_f * t5 * t20 * t29
    t758 = F_s_mag * m_s * t5 * t23 * t29
    t761 = t8 ** 2 * t118 * 4.0
    t772 = t18 * t118 * 4.0
    t780 = K_p * t57 * t118 * -2.0
    t781 = -t473
    t782 = m_s * t6 * t10 * t51 * -2.0
    t784 = -t476
    t786 = K_p * t59 * t118 * -2.0
    t794 = F_s_mag * t3 * t18 * t71
    t795 = t46 * t133
    t800 = -t493
    t801 = -t494
    t802 = -t495
    t803 = -t496
    t804 = -t497
    t809 = -t502
    t811 = -t506
    t825 = t7 + t9 + t34

    t827 = -t228
    t828 = F_f_mag * t2 * t19 * t49
    t831 = -t542
    t837 = F_s_mag * t5 * t17 * t49
    t838 = -t236
    t851 = F_f_mag * t4 * t41 * t49
    t855 = t8 * t46 * t81

    mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s = ft_1([
        F_f_mag, F_s_mag, K_f, K_p, K_s, m_c, m_f, m_s, t10, t100, t101, t103, t104, t105, t106, t107, t108, t109,
        t111, t112, t113, t115, t116, t117, t118, t119, t120, t121, t122, t123, t124, t1243, t126, t129, t131, t132,
        t133, t134, t135, t136, t141, t143, t145, t146, t149, t150, t152, t154, t155, t157, t158, t159, t160, t161,
        t163, t165, t166, t168, t169, t170, t172, t173, t174, t175, t176, t178, t18, t183, t184, t185, t186, t187,
        t188, t189, t19, t191, t194, t195, t196, t197, t198, t199, t2, t20, t201, t204, t205, t207, t208, t211, t212,
        t213, t216, t217, t218, t219, t22, t221, t222, t223, t224, t225, t226, t228, t229, t23, t230, t231, t232, t233,
        t234, t235, t236, t237, t238, t24, t242, t243, t247, t248, t249, t25, t250, t251, t252, t254, t256, t258, t259,
        t26, t260, t261, t262, t264, t267, t268, t27, t270, t272, t273, t274, t275, t276, t278, t283, t287, t292, t296,
        t297, t299, t3, t300, t304, t307, t31, t310, t312, t315, t316, t32, t320, t323, t328, t33, t332, t337, t34,
        t341,
        t343, t345, t347, t349, t35, t351, t352, t356, t359, t36, t362, t364, t367, t368, t372, t379, t381, t383, t385,
        t390, t391, t393, t394, t395, t396, t399, t4, t400, t401, t403, t404, t406, t41, t411, t412, t414, t415, t416,
        t417, t42, t420, t421, t422, t425, t428, t429, t43, t430, t431, t432, t433, t434, t435, t436, t437, t438, t439,
        t44, t440, t441, t442, t443, t444, t448, t45, t451, t452, t453, t454, t455, t456, t457, t458, t459, t46, t460,
        t461, t462, t463, t469, t47, t470, t471, t476, t477, t479, t48, t481, t485, t49, t495, t499, t5, t50, t500,
        t501,
        t504, t505, t507, t51, t510, t511, t512, t516, t517, t518, t519, t520, t521, t522, t523, t524, t525, t526, t527,
        t528, t529, t53, t532, t534, t536, t538, t539, t54, t540, t541, t543, t544, t546, t547, t548, t549, t55, t550,
        t552, t553, t554, t555, t556, t56, t560, t562, t563, t564, t565, t567, t57, t571, t573, t575, t576, t577, t58,
        t580, t584, t59, t6, t611, t614, t616, t617, t618, t63, t631, t633, t634, t638, t64, t641, t643, t644, t645,
        t647, t65, t650, t651, t653, t656, t658, t66, t661, t663, t664, t667, t668, t669, t67, t670, t672, t673, t674,
        t68, t681, t682, t683, t684, t685, t69, t691, t692, t694, t696, t698, t7, t70, t700, t707, t709, t71, t710,
        t712, t713, t714, t715, t716, t717, t719, t72, t720, t721, t722, t723, t724, t726, t727, t728, t729, t73, t730,
        t731, t732, t733, t734, t74, t75, t755, t756, t757, t758, t76, t761, t77, t772, t78, t780, t781, t782, t784,
        t786, t79, t794, t795, t8, t80, t800, t801, t802, t803, t804, t809, t81, t811, t82, t825, t827, t828, t831,
        t837, t838, t85, t851, t855, t86, t87, t88, t89, t9, t90, t91, t93, t94, t95, t96, t98, t99, w
    ])

    return mag_U,phase_U,real_P,imag_P,mag_X_u,phase_X_u,mag_X_f,phase_X_f,mag_X_s,phase_X_s


def ft_1(ct):
    mag_U = 0
    phase_U = 0
    real_P = 0
    imag_P = 0
    mag_X_u = 0
    phase_X_u = 0
    mag_X_f = 0
    phase_X_f = 0
    mag_X_s = 0
    phase_X_s = 0

    F_f_mag = ct[0]
    F_s_mag = ct[1]
    K_f = ct[2]
    K_p = ct[3]
    K_s = ct[4]
    m_c = ct[5]
    m_f = ct[6]
    m_s = ct[7]
    t10 = ct[8]
    t100 = ct[9]
    t101 = ct[10]
    t103 = ct[11]
    t104 = ct[12]
    t105 = ct[13]
    t106 = ct[14]
    t107 = ct[15]
    t108 = ct[16]
    t109 = ct[17]
    t111 = ct[18]
    t112 = ct[19]
    t113 = ct[20]
    t115 = ct[21]
    t116 = ct[22]
    t117 = ct[23]
    t118 = ct[24]
    t119 = ct[25]
    t120 = ct[26]
    t121 = ct[27]
    t122 = ct[28]
    t123 = ct[29]
    t124 = ct[30]
    t1243 = ct[31]
    t126 = ct[32]
    t129 = ct[33]
    t131 = ct[34]
    t132 = ct[35]
    t133 = ct[36]
    t134 = ct[37]
    t135 = ct[38]
    t136 = ct[39]
    t141 = ct[40]
    t143 = ct[41]
    t145 = ct[42]
    t146 = ct[43]
    t149 = ct[44]
    t150 = ct[45]
    t152 = ct[46]
    t154 = ct[47]
    t155 = ct[48]
    t157 = ct[49]
    t158 = ct[50]
    t159 = ct[51]
    t160 = ct[52]
    t161 = ct[53]
    t163 = ct[54]
    t165 = ct[55]
    t166 = ct[56]
    t168 = ct[57]
    t169 = ct[58]
    t170 = ct[59]
    t172 = ct[60]
    t173 = ct[61]
    t174 = ct[62]
    t175 = ct[63]
    t176 = ct[64]
    t178 = ct[65]
    t18 = ct[66]
    t183 = ct[67]
    t184 = ct[68]
    t185 = ct[69]
    t186 = ct[70]
    t187 = ct[71]
    t188 = ct[72]
    t189 = ct[73]
    t19 = ct[74]
    t191 = ct[75]
    t194 = ct[76]
    t195 = ct[77]
    t196 = ct[78]
    t197 = ct[79]

    t198 = ct[80]
    t199 = ct[81]
    t2 = ct[82]
    t20 = ct[83]
    t201 = ct[84]
    t204 = ct[85]
    t205 = ct[86]
    t207 = ct[87]
    t208 = ct[88]
    t211 = ct[89]
    t212 = ct[90]
    t213 = ct[91]
    t216 = ct[92]
    t217 = ct[93]
    t218 = ct[94]
    t219 = ct[95]
    t22 = ct[96]
    t221 = ct[97]
    t222 = ct[98]
    t223 = ct[99]
    t224 = ct[100]
    t225 = ct[101]
    t226 = ct[102]
    t228 = ct[103]
    t229 = ct[104]
    t23 = ct[105]
    t230 = ct[106]
    t231 = ct[107]
    t232 = ct[108]
    t233 = ct[109]
    t234 = ct[110]
    t235 = ct[111]
    t236 = ct[112]
    t237 = ct[113]
    t238 = ct[114]
    t24 = ct[115]
    t242 = ct[116]
    t243 = ct[117]
    t247 = ct[118]
    t248 = ct[119]
    t249 = ct[120]
    t25 = ct[121]
    t250 = ct[122]
    t251 = ct[123]
    t252 = ct[124]
    t254 = ct[125]
    t256 = ct[126]
    t258 = ct[127]
    t259 = ct[128]
    t26 = ct[129]
    t260 = ct[130]
    t261 = ct[131]
    t262 = ct[132]
    t264 = ct[133]
    t267 = ct[134]
    t268 = ct[135]
    t27 = ct[136]
    t270 = ct[137]
    t272 = ct[138]
    t273 = ct[139]
    t274 = ct[140]
    t275 = ct[141]
    t276 = ct[142]
    t278 = ct[143]
    t283 = ct[144]
    t287 = ct[145]
    t292 = ct[146]
    t296 = ct[147]
    t297 = ct[148]
    t299 = ct[149]
    t3 = ct[150]
    t300 = ct[151]
    t304 = ct[152]
    t307 = ct[153]
    t31 = ct[154]
    t310 = ct[155]
    t312 = ct[156]
    t315 = ct[157]
    t316 = ct[158]
    t32 = ct[159]
    t320 = ct[160]
    t323 = ct[161]
    t328 = ct[162]
    t33 = ct[163]
    t332 = ct[164]
    t337 = ct[165]
    t34 = ct[166]
    t341 = ct[167]
    t343 = ct[168]
    t345 = ct[169]
    t347 = ct[170]
    t349 = ct[171]
    t35 = ct[172]
    t351 = ct[173]
    t352 = ct[174]
    t356 = ct[175]
    t359 = ct[176]
    t36 = ct[177]
    t362 = ct[178]
    t364 = ct[179]

    t367 = ct[180]
    t368 = ct[181]
    t372 = ct[182]
    t379 = ct[183]
    t381 = ct[184]
    t383 = ct[185]
    t385 = ct[186]
    t390 = ct[187]
    t391 = ct[188]
    t393 = ct[189]
    t394 = ct[190]
    t395 = ct[191]
    t396 = ct[192]
    t399 = ct[193]
    t4 = ct[194]
    t400 = ct[195]
    t401 = ct[196]
    t403 = ct[197]
    t404 = ct[198]
    t406 = ct[199]
    t41 = ct[200]
    t411 = ct[201]
    t412 = ct[202]
    t414 = ct[203]
    t415 = ct[204]
    t416 = ct[205]
    t417 = ct[206]
    t42 = ct[207]
    t420 = ct[208]
    t421 = ct[209]
    t422 = ct[210]
    t425 = ct[211]
    t428 = ct[212]
    t429 = ct[213]
    t43 = ct[214]
    t430 = ct[215]
    t431 = ct[216]
    t432 = ct[217]
    t433 = ct[218]
    t434 = ct[219]
    t435 = ct[220]
    t436 = ct[221]
    t437 = ct[222]
    t438 = ct[223]
    t439 = ct[224]
    t44 = ct[225]
    t440 = ct[226]
    t441 = ct[227]
    t442 = ct[228]
    t443 = ct[229]
    t444 = ct[230]
    t448 = ct[231]
    t45 = ct[232]
    t451 = ct[233]
    t452 = ct[234]
    t453 = ct[235]
    t454 = ct[236]
    t455 = ct[237]
    t456 = ct[238]
    t457 = ct[239]
    t458 = ct[240]
    t459 = ct[241]
    t46 = ct[242]
    t460 = ct[243]
    t461 = ct[244]
    t462 = ct[245]
    t463 = ct[246]
    t469 = ct[247]
    t47 = ct[248]
    t470 = ct[249]
    t471 = ct[250]
    t476 = ct[251]
    t477 = ct[252]
    t479 = ct[253]
    t48 = ct[254]
    t481 = ct[255]
    t485 = ct[256]
    t49 = ct[257]
    t495 = ct[258]
    t499 = ct[259]
    t5 = ct[260]
    t50 = ct[261]
    t500 = ct[262]
    t501 = ct[263]
    t504 = ct[264]
    t505 = ct[265]
    t507 = ct[266]
    t51 = ct[267]
    t510 = ct[268]
    t511 = ct[269]
    t512 = ct[270]
    t516 = ct[271]
    t517 = ct[272]
    t518 = ct[273]
    t519 = ct[274]
    t520 = ct[275]
    t521 = ct[276]
    t522 = ct[277]
    t523 = ct[278]
    t524 = ct[279]

    t525 = ct[280]
    t526 = ct[281]
    t527 = ct[282]
    t528 = ct[283]
    t529 = ct[284]
    t53 = ct[285]
    t532 = ct[286]
    t534 = ct[287]
    t536 = ct[288]
    t538 = ct[289]
    t539 = ct[290]
    t54 = ct[291]
    t540 = ct[292]
    t541 = ct[293]
    t543 = ct[294]
    t544 = ct[295]
    t546 = ct[296]
    t547 = ct[297]
    t548 = ct[298]
    t549 = ct[299]
    t55 = ct[300]
    t550 = ct[301]
    t552 = ct[302]
    t553 = ct[303]
    t554 = ct[304]
    t555 = ct[305]
    t556 = ct[306]
    t56 = ct[307]
    t560 = ct[308]
    t562 = ct[309]
    t563 = ct[310]
    t564 = ct[311]
    t565 = ct[312]
    t567 = ct[313]
    t57 = ct[314]
    t571 = ct[315]
    t573 = ct[316]
    t575 = ct[317]
    t576 = ct[318]
    t577 = ct[319]
    t58 = ct[320]
    t580 = ct[321]
    t584 = ct[322]
    t59 = ct[323]
    t6 = ct[324]
    t611 = ct[325]
    t614 = ct[326]
    t616 = ct[327]
    t617 = ct[328]
    t618 = ct[329]
    t63 = ct[330]
    t631 = ct[331]
    t633 = ct[332]
    t634 = ct[333]
    t638 = ct[334]
    t64 = ct[335]
    t641 = ct[336]
    t643 = ct[337]
    t644 = ct[338]
    t645 = ct[339]
    t647 = ct[340]
    t65 = ct[341]
    t650 = ct[342]
    t651 = ct[343]
    t653 = ct[344]
    t656 = ct[345]
    t658 = ct[346]
    t66 = ct[347]
    t661 = ct[348]
    t663 = ct[349]
    t664 = ct[350]
    t667 = ct[351]
    t668 = ct[352]
    t669 = ct[353]
    t67 = ct[354]
    t670 = ct[355]
    t672 = ct[356]
    t673 = ct[357]
    t674 = ct[358]
    t68 = ct[359]
    t681 = ct[360]
    t682 = ct[361]
    t683 = ct[362]
    t684 = ct[363]
    t685 = ct[364]
    t69 = ct[365]
    t691 = ct[366]
    t692 = ct[367]
    t694 = ct[368]
    t696 = ct[369]
    t698 = ct[370]
    t7 = ct[371]
    t70 = ct[372]
    t700 = ct[373]
    t707 = ct[374]
    t709 = ct[375]
    t71 = ct[376]
    t710 = ct[377]
    t712 = ct[378]
    t713 = ct[379]

    t714 = ct[380]
    t715 = ct[381]
    t716 = ct[382]
    t717 = ct[383]
    t719 = ct[384]
    t72 = ct[385]
    t720 = ct[386]
    t721 = ct[387]
    t722 = ct[388]
    t723 = ct[389]
    t724 = ct[390]
    t726 = ct[391]
    t727 = ct[392]
    t728 = ct[393]
    t729 = ct[394]
    t73 = ct[395]
    t730 = ct[396]
    t731 = ct[397]
    t732 = ct[398]
    t733 = ct[399]
    t734 = ct[400]
    t74 = ct[401]
    t75 = ct[402]
    t755 = ct[403]
    t756 = ct[404]
    t757 = ct[405]
    t758 = ct[406]
    t76 = ct[407]
    t761 = ct[408]
    t77 = ct[409]
    t772 = ct[410]
    t78 = ct[411]
    t780 = ct[412]
    t781 = ct[413]
    t782 = ct[414]
    t784 = ct[415]
    t786 = ct[416]
    t79 = ct[417]
    t794 = ct[418]
    t795 = ct[419]
    t8 = ct[420]
    t80 = ct[421]
    t800 = ct[422]
    t801 = ct[423]
    t802 = ct[424]
    t803 = ct[425]
    t804 = ct[426]
    t809 = ct[427]
    t81 = ct[428]
    t811 = ct[429]
    t82 = ct[430]
    t825 = ct[431]
    t827 = ct[432]
    t828 = ct[433]
    t831 = ct[434]
    t837 = ct[435]
    t838 = ct[436]
    t85 = ct[437]
    t851 = ct[438]
    t855 = ct[439]
    t86 = ct[440]
    t87 = ct[441]
    t88 = ct[442]
    t89 = ct[443]
    t9 = ct[444]
    t90 = ct[445]
    t91 = ct[446]
    t93 = ct[447]
    t94 = ct[448]
    t95 = ct[449]
    t96 = ct[450]
    t98 = ct[451]
    t99 = ct[452]
    w = ct[453]

    t857 = F_s_mag * t5 * t41 * t49
    t859 = t8 * t34 * t74
    t860 = t8 * t34 * t76
    t861 = t18 * t45 * t79
    t862 = t45 * t234
    t867 = K_p * t34 * t74
    t868 = K_p * t34 * t76
    t871 = t49 * t57 * t74
    t877 = t49 * t57 * t76
    t881 = t8 * t34 * t79
    t882 = t8 * t34 * t81
    t885 = F_f_mag * m_s * t2 * t50 * t51
    t892 = F_s_mag * t9 * t50 * t70
    t893 = -t616
    t897 = F_s_mag * t3 * t49 * t95
    t900 = K_p * t34 * t79
    t901 = K_p * t34 * t81
    t902 = t49 * t57 * t79
    t908 = t49 * t57 * t81
    t912 = F_f_mag * K_p * t50 * t69
    t915 = F_f_mag * K_p * m_s * t2 * t50 * w
    t920 = F_f_mag * t10 * t49 * t72
    t924 = F_s_mag * m_c * t5 * t10 * t51 * -2.0
    t927 = F_f_mag * t4 * t49 * t95
    t931 = F_s_mag * m_s * t5 * t50 * t51
    t932 = t46 * t57 * t81

    t938 = F_f_mag * K_p * t50 * t72
    t941 = F_f_mag * K_p * m_s * t4 * t50 * w
    t947 = F_f_mag * m_c * m_f * t2 * t27 * t49
    t950 = -2.0 * m_c * m_f * t27 * t76
    t957 = F_s_mag * m_s * t3 * t50 * t56
    t958 = F_s_mag * m_c * m_f * t3 * t27 * t46
    t972 = -2.0 * m_c * m_s * t27 * t79
    t977 = F_s_mag * m_c * m_s * t5 * t27 * t49
    t979 = t7 * t18 * t74
    t980 = t8 * t228
    t982 = t9 * t18 * t76
    t984 = t9 * t230
    t986 = F_s_mag * m_s * t5 * t50 * t56
    t987 = F_s_mag * m_c * m_f * t5 * t27 * t46
    t992 = t8 * t149
    t996 = t8 * t155
    t1023 = K_s * t633
    t1034 = K_s * t643
    t1069 = F_f_mag * t10 * t18 * t69
    t1071 = F_f_mag * t2 * t18 * t95
    t1072 = t57 * t132
    t1074 = K_p * t668
    t1075 = K_p * t670
    t1077 = F_s_mag * t3 * t18 * t95
    t1078 = K_p * t673
    t1082 = t18 * t57 * t79
    t1084 = t18 * t59 * t79
    t1089 = t57 * t234
    t1092 = t8 * t713
    t1094 = t18 * t57 * t81
    t1097 = t8 * t715
    t1098 = t18 * t59 * t81
    t1101 = t8 * t717
    t1118 = F_f_mag * m_c * m_f * t2 * t18 * t27
    t1123 = F_s_mag * m_c * m_s * t3 * t18 * t27
    t1133 = t7 * t49 * t74
    t1143 = t8 * t46 * t76
    t1145 = F_s_mag * t3 * t41 * t49
    t1149 = t9 * t49 * t81
    t1152 = -2.0 * t6 * t8 * t74
    t1158 = -2.0 * K_p * t6 * t76
    t1162 = -2.0 * t6 * t8 * t81
    t1167 = -2.0 * K_p * t6 * t81
    t1178 = F_f_mag * t2 * t24 * t27 * t49
    t1184 = F_s_mag * t3 * t23 * t27 * t46
    t1195 = F_s_mag * t5 * t23 * t27 * t49
    t1206 = F_s_mag * t5 * t23 * t27 * t46
    t1217 = -2.0 * F_s_mag * m_c * t3 * t8 * t10 * t51
    t1230 = t8 * t229
    t1233 = t8 * t230
    t1242 = F_s_mag * t8 * t10 * t46 * t73
    t1247 = F_s_mag * t3 * t476
    t1248 = -2.0 * m_c * m_s * t8 * t27 * t74
    t1253 = 3.0 * F_s_mag * t10 * t18 * t70
    t1255 = F_s_mag * t3 * t495
    t1260 = -2.0 * F_s_mag * K_p * m_s * t3 * t10 * t58
    t1272 = 3.0 * F_f_mag * m_c * m_s * t2 * t18 * t27
    t1273 = 3.0 * F_s_mag * m_c * m_f * t3 * t18 * t27
    t1285 = t49 * t231
    t1289 = t49 * t155
    t1290 = t8 ** 2 * t45 * t79
    t1301 = t18 * t34 * t74
    t1302 = t18 * t34 * t76
    t1323 = F_f_mag * t18 * t50 * t69
    t1326 = F_f_mag * m_s * t2 * t18 * t50 * w
    t1335 = t49 * t713
    t1338 = t49 * t715
    t1341 = t49 * t717
    t1347 = -3.0 * F_s_mag * m_c * m_f * t3 * t8 ** 2 * t27
    t1362 = t46 + t55 + t59
    t1363 = -2.0 * t6 * t8 ** 2 * t76
    t1365 = -2.0 * t6 * t18 * t76
    t1367 = t49 * t230
    t1376 = t50 + t56 + t58
    t110 = -t54
    t114 = -t64
    t125 = t22 * t48
    t127 = -t77
    t128 = -t78
    t130 = -t80
    t139 = -t107
    t140 = -t108
    t142 = F_f_mag * t104
    t144 = F_s_mag * t103
    t147 = K_p * t6 * t53
    t148 = K_p * t7 * t53
    t151 = K_p * t6 * t54
    t156 = K_p * t9 * t54
    t162 = F_f_mag * t106
    t164 = F_s_mag * t105
    t167 = K_p * t6 * t63
    t171 = K_p * t6 * t64
    t177 = -t91
    t179 = -t93
    t180 = -t94
    t181 = t78 * 1j  # MATLAB's 1i -> Python's 1j for imaginary
    t192 = -t105
    t193 = -t106
    t200 = 4.0 * K_p * t6 * t93
    t202 = 4.0 * t6 * t8 * t96

    t203 = 4.0 * K_p * t6 * t94
    t220 = 4.0 * t6 * t8 * t78
    t227 = 4.0 * K_p * K_s * t78
    t239 = t6 * t158
    t240 = t93 * 1j
    t241 = t94 * 1j
    t244 = K_p * t166
    t245 = K_p * t172
    t246 = t98 * 1j
    t253 = t6 * t8 * t63
    t255 = K_p * t7 * t80
    t257 = t6 * t8 * t64
    t263 = K_p * t9 * t82
    t266 = t6 * t18 * t63
    t271 = t6 * t18 * t64
    t279 = t7 * t8 * t141
    t280 = t74 * t93
    t281 = t6 * t57 * t75
    t282 = t7 * t55 * t75
    t284 = t8 * t55 * t75
    t285 = F_f_mag * t2 * t8 * t93
    t286 = t8 * t57 * t75
    t288 = t8 * t55 * t77
    t289 = t6 * t59 * t77
    t290 = t76 * t94
    t291 = t9 * t55 * t77
    t293 = t8 * t9 * t143
    t294 = t8 * t59 * t77
    t295 = F_s_mag * t3 * t8 * t94
    t298 = K_p * t7 * t141
    t301 = t6 * t53 * t57
    t302 = t7 * t53 * t55
    t303 = m_c * t10 * t75 * w
    t305 = K_p * t55 * t75
    t306 = F_f_mag * K_p * t2 * t93
    t308 = t8 * t53 * t57
    t309 = K_p * t57 * t75
    t311 = K_p * t55 * t77
    t313 = t76 * t96
    t314 = t41 * t143
    t317 = K_s * t8 * t143
    t318 = K_p * t9 * t143
    t319 = K_p * t59 * t77
    t321 = F_s_mag * t3 * t8 * t96
    t322 = F_s_mag * K_p * t3 * t94
    t324 = t7 * t8 * t161
    t325 = t79 * t93
    t326 = t6 * t57 * t80
    t327 = t7 * t55 * t80
    t329 = t8 * t55 * t80
    t330 = F_f_mag * t4 * t8 * t93
    t331 = t8 * t57 * t80
    t333 = t8 * t55 * t82
    t334 = t6 * t59 * t82
    t335 = t81 * t94
    t336 = t9 * t55 * t82
    t338 = t8 * t9 * t163
    t339 = t8 * t59 * t82
    t340 = F_s_mag * t5 * t8 * t94
    t342 = m_c * t10 * t53 * w
    t344 = K_p * t53 * t57
    t346 = K_p * K_s * t143
    t348 = F_s_mag * K_p * t3 * t96
    t350 = K_p * t7 * t161
    t353 = t6 * t57 * t63
    t354 = t41 * t161
    t355 = m_c * t10 * t80 * w
    t357 = K_p * t55 * t80
    t358 = F_f_mag * K_p * t4 * t93
    t360 = t8 * t57 * t63
    t361 = K_p * t57 * t80
    t363 = K_p * t55 * t82
    t365 = t81 * t96
    t366 = t41 * t163
    t369 = K_s * t8 * t163
    t370 = K_p * t9 * t163
    t371 = K_p * t59 * t82
    t373 = F_s_mag * t5 * t8 * t96
    t374 = F_s_mag * K_p * t5 * t94
    t376 = m_c * t10 * t63 * w
    t378 = K_p * t57 * t63
    t380 = K_p * K_s * t163
    t382 = F_s_mag * K_p * t5 * t96
    t384 = t74 * t78
    t386 = m_c * m_f * t27 * t75
    t387 = F_f_mag * t2 * t8 * t78
    t388 = t76 * t78
    t389 = m_c * m_s * t27 * t77
    t392 = F_s_mag * t3 * t8 * t78
    t397 = m_c * m_f * t27 * t53
    t398 = F_f_mag * K_p * t2 * t78
    t402 = F_s_mag * K_p * t3 * t78
    t405 = t78 * t79
    t407 = m_c * m_f * t27 * t80
    t408 = F_f_mag * t4 * t8 * t78
    t409 = t78 * t81
    t410 = m_c * m_s * t27 * t82
    t413 = F_s_mag * t5 * t8 * t78
    t418 = m_c * m_f * t27 * t63
    t419 = F_f_mag * K_p * t4 * t78

    t423 = F_s_mag * K_p * t5 * t78
    t447 = -t204
    t449 = -t207
    t450 = -t208
    t464 = -t211
    t465 = 4.0 * t6 * t8 * t98
    t466 = 2.0 * t7 * t8 * t98
    t467 = 2.0 * t90 * t98
    t468 = 2.0 * t8 * t9 * t98
    t475 = t6 * t34 * t96
    t480 = 2.0 * K_p * t7 * t93
    t482 = 2.0 * t8 ** 2 * t96
    t483 = 2.0 * K_p * t9 * t94
    t484 = 2.0 * t41 * t93
    t486 = -t141
    t488 = -t216
    t489 = -t217
    t490 = -t218
    t491 = -t219
    t498 = 2.0 * K_p * K_s * t98
    t503 = 2.0 * t18 * t96
    t513 = -t154
    t514 = -t161
    t530 = -t170
    t533 = -t176
    t535 = t6 * t34 * t78
    t537 = 2.0 * t8 ** 2 * t78
    t545 = 2.0 * t18 * t78
    t551 = -t237
    t557 = t6 * t7 * t75
    t558 = t7 * t8 * t75
    t559 = t6 * t9 * t77
    t561 = t8 * t9 * t77
    t566 = t6 * t7 * t53
    t568 = t6 * t8 * t53
    t569 = t7 * t8 * t53
    t570 = K_p * t7 * t75
    t572 = t6 * t8 * t54
    t574 = t6 * t9 * t54
    t578 = K_p * t9 * t77
    t579 = t8 * t9 * t54
    t581 = t6 * t7 * t80
    t582 = t7 * t8 * t80
    t583 = t6 * t9 * t82
    t585 = t8 * t9 * t82
    t586 = 3.0 * K_p * t6 * t80
    t587 = 3.0 * K_p * t6 * t82
    t588 = t8 * t549
    t589 = t75 * t118
    t590 = t75 * t90
    t591 = t8 * t9 * t75
    t592 = t77 * t118
    t593 = t8 * t550
    t594 = t7 * t8 * t77
    t595 = t77 * t90
    t596 = t53 * t118
    t597 = t42 * t75
    t598 = t54 * t118
    t599 = K_p * t9 * t75
    t600 = K_p * t7 * t77
    t601 = t41 * t77
    t602 = t75 * t121
    t603 = t77 * t121
    t604 = t80 * t118
    t605 = t80 * t90
    t606 = t8 * t9 * t80
    t607 = t82 * t118
    t608 = t7 * t8 * t82
    t609 = t82 * t90
    t610 = t53 * t121
    t612 = 3.0 * t8 * t53 * t55
    t613 = 2.0 * t53 * t93
    t615 = 2.0 * t8 * t53 * t59
    t619 = K_p * t553
    t620 = t63 * t118
    t621 = t42 * t80
    t622 = t64 * t118
    t623 = K_p * t9 * t80
    t624 = K_p * t554
    t625 = K_p * t7 * t82
    t626 = t41 * t82
    t627 = t80 * t121
    t628 = t82 * t121
    t629 = 3.0 * K_p * t53 * t55
    t630 = 2.0 * t53 * t95
    t632 = 2.0 * K_p * t53 * t59
    t637 = t63 * t121
    t639 = 3.0 * t8 * t55 * t63
    t642 = 2.0 * t8 * t59 * t63
    t646 = t74 * t98
    t648 = t75 * t98
    t649 = t75 * t94
    t652 = t76 * t98
    t654 = t77 * t98
    t657 = t77 * t93
    t659 = 3.0 * K_p * t55 * t63
    t660 = 2.0 * t63 * t95
    t662 = 2.0 * K_p * t59 * t63
    t665 = K_p * t562
    t666 = t53 * t98

    t671 = K_p * t563
    t676 = t8 * t146
    t677 = K_p * t567
    t678 = K_p * t573
    t679 = t8 * t152
    t686 = t79 * t98
    t687 = t8 * t247
    t689 = t80 * t98
    t690 = t80 * t94
    t693 = t81 * t98
    t695 = t82 * t98
    t697 = t8 * t248
    t699 = t82 * t93
    t701 = t8 * t251
    t703 = t8 * t252
    t704 = t8 * t258
    t706 = t8 * t259
    t708 = 2.0 * t53 * t78
    t711 = t63 * t98
    t725 = 2.0 * t63 * t78
    t736 = t8 * t300
    t743 = t8 * t312
    t748 = t8 * t315
    t760 = K_p * t345
    t762 = K_p * t352
    t767 = K_p * t364
    t770 = t8 * t379
    t771 = K_p * t367
    t774 = t8 * t385
    t778 = t8 * t390
    t783 = -2.0 * t96 * t118
    t785 = -t477
    t787 = -t479
    t789 = -t481
    t793 = -t485
    t796 = K_p * t396
    t798 = K_p * t399
    t799 = K_p * t400
    t806 = -t499
    t807 = -t500
    t808 = -t501
    t812 = -t507
    t814 = K_p * t406
    t815 = t8 * t417
    t820 = t8 * t420
    t821 = t8 * t421
    t822 = K_p * t411
    t823 = t6 * t34 * t98
    t824 = 4.0 * t8 ** 2 * t98
    t826 = 4.0 * t18 * t98
    t829 = K_p * t437
    t830 = K_p * t438
    t832 = -t546
    t833 = -t547
    t834 = -t548
    t835 = t8 * t454
    t836 = t8 * t455
    t839 = t23 * t24 * t48
    t840 = -t552
    t841 = 2.0 * m_f * m_s * t20 * t48
    t842 = 3.0 * t6 * t8 * t75
    t843 = 3.0 * t6 * t8 * t77
    t844 = 3.0 * K_p * t6 * t75
    t845 = 3.0 * K_p * t6 * t77
    t846 = K_p * t529
    t847 = 3.0 * t6 * t8 * t80
    t848 = 3.0 * t6 * t8 * t82
    t849 = -t248
    t850 = t6 * t49 * t63
    t853 = -t258
    t854 = -t259
    t856 = -t261
    t863 = -t270
    t865 = -t275
    t866 = -t276
    t869 = F_f_mag * K_s * t183
    t870 = F_s_mag * K_f * t184
    t874 = t49 * t55 * t75
    t875 = -2.0 * t6 * t55 * t77
    t876 = t49 * t57 * t75
    t879 = t9 * t49 * t143
    t880 = F_s_mag * t3 * t49 * t94
    t883 = -t297
    t884 = -2.0 * t6 * t53 * t55
    t889 = K_p * t57 * t126
    t890 = -t614
    t894 = -t617
    t895 = t8 * t46 * t143
    t903 = t93 * t129
    t905 = t49 * t55 * t80
    t906 = -2.0 * t6 * t55 * t82
    t907 = t49 * t57 * t80
    t910 = t9 * t49 * t163
    t911 = F_s_mag * t5 * t49 * t94
    t913 = m_c * t50 * t53 * w
    t922 = m_c * t50 * t80 * w
    t923 = K_p * t59 * t129
    t928 = t49 * t57 * t63
    t934 = -t368
    t936 = 2.0 * F_f_mag * t2 * t8 * t98

    t937 = 2.0 * F_s_mag * t3 * t8 * t98
    t939 = m_c * t50 * t63 * w
    t945 = 2.0 * F_f_mag * K_p * t2 * t98
    t946 = 2.0 * F_s_mag * K_p * t3 * t98
    t948 = -t385
    t951 = -t683
    t952 = F_f_mag * t2 * t49 * t78
    t953 = t78 * t126
    t954 = 2.0 * F_f_mag * t4 * t8 * t98
    t955 = 2.0 * F_s_mag * t5 * t8 * t98
    t956 = -t709
    t959 = -t710
    t960 = -t401
    t962 = t8 * t556
    t967 = t8 * t560
    t970 = 2.0 * F_f_mag * K_p * t4 * t98
    t971 = 2.0 * F_s_mag * K_p * t5 * t98
    t974 = -t723
    t976 = -t411
    t978 = F_s_mag * t5 * t49 * t78
    t981 = t7 * t18 * t75
    t983 = t9 * t18 * t77
    t985 = -t726
    t988 = -t727
    t989 = -t422
    t1000 = t8 * t611
    t1002 = t8 * t617
    t1003 = t8 * t250
    t1007 = t8 * t254
    t1010 = t8 * t256
    t1014 = t8 * t260
    t1015 = t8 * t262
    t1018 = t8 * t264
    t1021 = -t437
    t1026 = t8 * t650
    t1029 = t9 * t656
    t1031 = K_p * t638
    t1035 = K_p * t644
    t1036 = -t454
    t1037 = K_p * t645
    t1038 = t8 * t299
    t1039 = t6 * t304
    t1040 = t8 * t307
    t1043 = K_p * t651
    t1044 = t6 * t310
    t1047 = t8 * t320
    t1049 = t8 * t681
    t1050 = t8 * t682
    t1051 = t8 * t685
    t1052 = t8 * t323
    t1054 = t8 * t328
    t1055 = t6 * t328
    t1060 = t8 * t692
    t1061 = t8 * t332
    t1063 = t6 * t332
    t1064 = t8 * t337
    t1073 = t18 * t53 * t57
    t1076 = K_s * t18 * t143
    t1079 = F_s_mag * t3 * t18 * t96
    t1080 = K_s * t672
    t1083 = t7 * t18 * t161
    t1085 = t18 * t55 * t80
    t1088 = F_f_mag * t4 * t18 * t93
    t1090 = t18 * t57 * t80
    t1091 = K_p * t691
    t1095 = t18 * t55 * t82
    t1099 = t9 * t18 * t163
    t1100 = t18 * t59 * t82
    t1102 = F_s_mag * t5 * t18 * t94
    t1103 = K_s * t698
    t1104 = K_p * t700
    t1106 = K_p * t710
    t1107 = K_p * t720
    t1108 = K_p * t721

    (mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s) = ft_2([
        F_f_mag, F_s_mag, K_p, m_c, m_f, m_s, t10, t100, t1002, t1007, t101, t1014, t1015, t1021, t1023,
        t1026, t103, t1035, t1036, t1037, t1039, t104, t1044, t1047, t1050, t1052, t1055, t1060, t1061,
        t107, t1072, t1073, t1074, t1075, t1076, t1077, t1078, t1079, t108, t1082, t1085, t1089, t109,
        t1090, t1091, t1094, t1095, t1098, t1099, t110, t1100, t1102, t1103, t1104, t1108, t111, t1118,
        t112, t1123, t113, t1133, t114, t1143, t1145, t1149, t115, t1152, t1158, t116, t1162, t1167,
        t117, t1178, t118, t1184, t119, t1195, t120, t1206, t121, t1217, t122, t123, t1230, t124, t1242,
        t1243, t1247, t1248, t125, t1253, t1255, t126, t1260, t127, t1272, t1273, t128, t1285, t1289,
        t129, t1290, t130, t1301, t131, t132, t1323, t1326, t133, t1335, t1338, t134, t1341, t1347,
        t135, t136, t1362, t1363, t1365, t1367, t1376, t139, t140, t141, t142, t143, t144, t145, t146,
        t147, t148, t150, t151, t155, t156, t157, t158, t159, t160, t162, t163, t164, t165, t166, t167,
        t168, t169, t170, t171, t172, t173, t174, t175, t176, t177, t178, t179, t18, t180, t181, t183,
        t184, t185, t186, t187, t188, t189, t19, t191, t192, t193, t194, t195, t196, t197, t198, t199,
        t2, t20, t200, t201, t202, t203, t205, t212, t213, t220, t221, t222, t223, t224, t225, t226,
        t227, t230, t231, t232, t233, t234, t235, t236, t238, t239, t24, t240, t241, t242, t243, t245,
        t246, t247, t249, t25, t250, t251, t252, t253, t255, t256, t257, t26, t263, t264, t266, t267,
        t268, t27, t272, t273, t274, t278, t279, t280, t281, t282, t283, t284, t285, t286, t287, t288,
        t289, t290, t291, t292, t293, t294, t295, t296, t298, t299, t3, t301, t302, t303, t304, t305,
        t306, t307, t308, t309, t31, t311, t312, t313, t314, t316, t317, t318, t319, t32, t321, t324,
        t325, t326, t327, t328, t329, t33, t330, t331, t333, t334, t335, t336, t337, t338, t339, t340,
        t341, t342, t343, t344, t345, t346, t347, t348, t349, t35, t350, t351, t352, t353, t354, t355,
        t356, t357, t358, t359, t36, t360, t361, t362, t363, t364, t365, t366, t367, t368, t369, t370,
        t371, t372, t373, t374, t376, t378, t379, t380, t381, t382, t383, t384, t386, t387, t388, t389,
        t390, t391, t392, t393, t394, t395, t396, t397, t398, t399, t4, t400, t401, t402, t403, t404,
        t405, t406, t407, t408, t409, t41, t410, t411, t412, t413, t414, t415, t416, t417, t418, t419,
        t42, t425, t428, t429, t43, t430, t431, t432, t433, t434, t435, t436, t437, t438, t439, t44,
        t440, t441, t442, t443, t444, t447, t448, t449, t45, t450, t451, t452, t453, t455, t456, t457,
        t458, t459, t46, t460, t461, t462, t463, t464, t465, t466, t467, t468, t469, t47, t470, t471,
        t480, t482, t483, t484, t486, t488, t489, t49, t490, t491, t498, t5, t50, t503, t504, t505, t51,
        t510, t511, t512, t513, t514, t516, t517, t518, t519, t520, t521, t522, t523, t524, t525, t526,
        t527, t528, t529, t53, t530, t532, t533, t534, t535, t536, t537, t538, t539, t54, t540, t541,
        t543, t544, t545, t549, t55, t551, t553, t554, t555, t557, t558, t559, t56, t560, t561, t562,
        t563, t564, t565, t566, t567, t568, t569, t57, t570, t571, t572, t573, t574, t575, t576, t577,
        t578, t579, t580, t581, t582, t583, t584, t585, t586, t587, t589, t59, t590, t591, t592, t593,
        t594, t595, t596, t597, t598, t599, t6, t600, t601, t602, t603, t604, t605, t606, t607, t608,
        t609, t610, t611, t612, t613, t614, t615, t618, t620, t621, t622, t623, t624, t625, t626, t627,
        t628, t629, t63, t630, t631, t632, t633, t634, t637, t638, t639, t641, t642, t643, t644, t645,
        t646, t647, t648, t649, t65, t650, t651, t652, t653, t654, t656, t657, t658, t659, t66, t660,
        t661, t662, t663, t664, t665, t666, t667, t668, t669, t67, t670, t671, t672, t673, t674, t677,
        t679, t68, t681, t683, t684, t685, t686, t689, t690, t691, t692, t693, t694, t695, t696, t697,
        t698, t699, t7, t70, t700, t704, t706, t707, t708, t709, t71, t710, t711, t712, t713, t714, t715,
        t716, t717, t719, t720, t721, t722, t723, t724, t725, t727, t728, t729, t73, t730, t731, t732,
        t733, t734, t736, t74, t748, t75, t755, t756, t757, t758, t76, t760, t761, t762, t77, t771,
        t772, t774, t78, t780, t781, t782, t783, t784, t785, t786, t787, t789, t79, t793, t794, t795,
        t796, t8, t80, t800, t801, t802, t803, t804, t806, t807, t808, t809, t81, t811, t812, t814, t82,
        t820, t821, t822, t823, t824, t825, t826, t827, t828, t829, t830, t831, t832, t833, t834, t835,
        t837, t838, t839, t840, t841, t842, t843, t844, t845, t846, t847, t848, t849, t85, t850, t851,
        t853, t854, t855, t856, t857, t859, t86, t860, t861, t862, t863, t865, t866, t867, t868, t869,
        t87, t871, t874, t875, t876, t877, t879, t88, t880, t881, t882, t883, t884, t885, t889, t89,
        t890, t892, t893, t894, t895, t897, t9, t90, t900, t902, t903, t905, t906, t907, t908, t910,
        t911, t912, t913, t915, t920, t922, t923, t924, t927, t928, t93, t931, t932, t934, t936, t937,
        t938, t939, t94, t941, t945, t946, t947, t948, t95, t950, t951, t952, t953, t954, t955, t956,
        t957, t958, t959, t96, t960, t962, t970, t971, t972, t974, t976, t977, t978, t979, t98, t980,
        t981, t985, t986, t987, t988, t989, t99, t992, w
    ])

    return mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s

def ft_2(ct):
    mag_U = 0
    phase_U = 0
    real_P = 0
    imag_P = 0
    mag_X_u = 0
    phase_X_u = 0
    mag_X_f = 0
    phase_X_f = 0
    mag_X_s = 0
    phase_X_s = 0

    # Unpack necessary variables from input list `ct`
    F_f_mag = ct[0]
    F_s_mag = ct[1]
    K_p = ct[2]
    m_c = ct[3]
    m_f = ct[4]
    m_s = ct[5]
    t10 = ct[6]
    t100 = ct[7]
    t1002 = ct[8]
    t1007 = ct[9]
    t101 = ct[10]
    t1014 = ct[11]
    t1015 = ct[12]

    t1021 = ct[13]
    t1023 = ct[14]
    t1026 = ct[15]
    t103 = ct[16]
    t1035 = ct[17]
    t1036 = ct[18]
    t1037 = ct[19]
    t1039 = ct[20]
    t104 = ct[21]
    t1044 = ct[22]
    t1047 = ct[23]
    t1050 = ct[24]
    t1052 = ct[25]
    t1055 = ct[26]
    t1060 = ct[27]
    t1061 = ct[28]
    t107 = ct[29]
    t1072 = ct[30]
    t1073 = ct[31]
    t1074 = ct[32]
    t1075 = ct[33]
    t1076 = ct[34]
    t1077 = ct[35]
    t1078 = ct[36]
    t1079 = ct[37]
    t108 = ct[38]
    t1082 = ct[39]
    t1085 = ct[40]
    t1089 = ct[41]
    t109 = ct[42]
    t1090 = ct[43]
    t1091 = ct[44]
    t1094 = ct[45]
    t1095 = ct[46]
    t1098 = ct[47]
    t1099 = ct[48]
    t110 = ct[49]
    t1100 = ct[50]
    t1102 = ct[51]
    t1103 = ct[52]
    t1104 = ct[53]
    t1108 = ct[54]
    t111 = ct[55]
    t1118 = ct[56]
    t112 = ct[57]
    t1123 = ct[58]
    t113 = ct[59]
    t1133 = ct[60]
    t114 = ct[61]
    t1143 = ct[62]
    t1145 = ct[63]
    t1149 = ct[64]
    t115 = ct[65]
    t1152 = ct[66]
    t1158 = ct[67]
    t116 = ct[68]
    t1162 = ct[69]
    t1167 = ct[70]
    t117 = ct[71]
    t1178 = ct[72]
    t118 = ct[73]
    t1184 = ct[74]
    t119 = ct[75]
    t1195 = ct[76]
    t120 = ct[77]
    t1206 = ct[78]
    t121 = ct[79]
    t1217 = ct[80]
    t122 = ct[81]
    t123 = ct[82]
    t1230 = ct[83]
    t124 = ct[84]
    t1242 = ct[85]
    t1243 = ct[86]
    t1247 = ct[87]
    t1248 = ct[88]
    t125 = ct[89]
    t1253 = ct[90]
    t1255 = ct[91]
    t126 = ct[92]
    t1260 = ct[93]
    t127 = ct[94]
    t1272 = ct[95]
    t1273 = ct[96]
    t128 = ct[97]
    t1285 = ct[98]
    t1289 = ct[99]
    t129 = ct[100]
    t1290 = ct[101]
    t130 = ct[102]
    t1301 = ct[103]
    t131 = ct[104]
    t132 = ct[105]
    t1323 = ct[106]
    t1326 = ct[107]
    t133 = ct[108]
    t1335 = ct[109]
    t1338 = ct[110]
    t134 = ct[111]
    t1341 = ct[112]

    t1347 = ct[113]
    t135 = ct[114]
    t136 = ct[115]
    t1362 = ct[116]
    t1363 = ct[117]
    t1365 = ct[118]
    t1367 = ct[119]
    t1376 = ct[120]
    t139 = ct[121]
    t140 = ct[122]
    t141 = ct[123]
    t142 = ct[124]
    t143 = ct[125]
    t144 = ct[126]
    t145 = ct[127]
    t146 = ct[128]
    t147 = ct[129]
    t148 = ct[130]
    t150 = ct[131]
    t151 = ct[132]
    t155 = ct[133]
    t156 = ct[134]
    t157 = ct[135]
    t158 = ct[136]
    t159 = ct[137]
    t160 = ct[138]
    t162 = ct[139]
    t163 = ct[140]
    t164 = ct[141]
    t165 = ct[142]
    t166 = ct[143]
    t167 = ct[144]
    t168 = ct[145]
    t169 = ct[146]
    t170 = ct[147]
    t171 = ct[148]
    t172 = ct[149]
    t173 = ct[150]
    t174 = ct[151]
    t175 = ct[152]
    t176 = ct[153]
    t177 = ct[154]
    t178 = ct[155]
    t179 = ct[156]
    t18 = ct[157]
    t180 = ct[158]
    t181 = ct[159]
    t183 = ct[160]
    t184 = ct[161]
    t185 = ct[162]
    t186 = ct[163]
    t187 = ct[164]
    t188 = ct[165]
    t189 = ct[166]
    t19 = ct[167]
    t191 = ct[168]
    t192 = ct[169]
    t193 = ct[170]
    t194 = ct[171]
    t195 = ct[172]
    t196 = ct[173]
    t197 = ct[174]
    t198 = ct[175]
    t199 = ct[176]
    t2 = ct[177]
    t20 = ct[178]
    t200 = ct[179]
    t201 = ct[180]
    t202 = ct[181]
    t203 = ct[182]
    t205 = ct[183]
    t212 = ct[184]
    t213 = ct[185]
    t220 = ct[186]
    t221 = ct[187]
    t222 = ct[188]
    t223 = ct[189]
    t224 = ct[190]
    t225 = ct[191]
    t226 = ct[192]
    t227 = ct[193]
    t230 = ct[194]
    t231 = ct[195]
    t232 = ct[196]
    t233 = ct[197]
    t234 = ct[198]
    t235 = ct[199]
    t236 = ct[200]
    t238 = ct[201]
    t239 = ct[202]
    t24 = ct[203]
    t240 = ct[204]
    t241 = ct[205]
    t242 = ct[206]
    t243 = ct[207]
    t245 = ct[208]
    t246 = ct[209]
    t247 = ct[210]
    t249 = ct[211]
    t25 = ct[212]

    t250 = ct[212]
    t251 = ct[213]
    t252 = ct[214]
    t253 = ct[215]
    t255 = ct[216]
    t256 = ct[217]
    t257 = ct[218]
    t26 = ct[219]
    t263 = ct[220]
    t264 = ct[221]
    t266 = ct[222]
    t267 = ct[223]
    t268 = ct[224]
    t27 = ct[225]
    t272 = ct[226]
    t273 = ct[227]
    t274 = ct[228]
    t278 = ct[229]
    t279 = ct[230]
    t280 = ct[231]
    t281 = ct[232]
    t282 = ct[233]
    t283 = ct[234]
    t284 = ct[235]
    t285 = ct[236]
    t286 = ct[237]
    t287 = ct[238]
    t288 = ct[239]
    t289 = ct[240]
    t290 = ct[241]
    t291 = ct[242]
    t292 = ct[243]
    t293 = ct[244]
    t294 = ct[245]
    t295 = ct[246]
    t296 = ct[247]
    t298 = ct[248]
    t299 = ct[249]
    t3 = ct[250]
    t301 = ct[251]
    t302 = ct[252]
    t303 = ct[253]
    t304 = ct[254]
    t305 = ct[255]
    t306 = ct[256]
    t307 = ct[257]
    t308 = ct[258]
    t309 = ct[259]
    t31 = ct[260]
    t311 = ct[261]
    t312 = ct[262]
    t313 = ct[263]
    t314 = ct[264]
    t316 = ct[265]
    t317 = ct[266]
    t318 = ct[267]
    t319 = ct[268]
    t32 = ct[269]
    t321 = ct[270]
    t324 = ct[271]
    t325 = ct[272]
    t326 = ct[273]
    t327 = ct[274]
    t328 = ct[275]
    t329 = ct[276]
    t33 = ct[277]
    t330 = ct[278]
    t331 = ct[279]
    t333 = ct[280]
    t334 = ct[281]
    t335 = ct[282]
    t336 = ct[283]
    t337 = ct[284]
    t338 = ct[285]
    t339 = ct[286]
    t340 = ct[287]
    t341 = ct[288]
    t342 = ct[289]
    t343 = ct[290]
    t344 = ct[291]
    t345 = ct[292]
    t346 = ct[293]
    t347 = ct[294]
    t348 = ct[295]
    t349 = ct[296]
    t35 = ct[297]
    t350 = ct[298]
    t351 = ct[299]
    t352 = ct[300]
    t353 = ct[301]
    t354 = ct[302]
    t355 = ct[303]
    t356 = ct[304]
    t357 = ct[305]
    t358 = ct[306]
    t359 = ct[307]
    t36 = ct[308]
    t360 = ct[309]
    t361 = ct[310]
    t362 = ct[311]

    t363 = ct[313]
    t364 = ct[314]
    t365 = ct[315]
    t366 = ct[316]
    t367 = ct[317]
    t368 = ct[318]
    t369 = ct[319]
    t370 = ct[320]
    t371 = ct[321]
    t372 = ct[322]
    t373 = ct[323]
    t374 = ct[324]
    t376 = ct[325]
    t378 = ct[326]
    t379 = ct[327]
    t380 = ct[328]
    t381 = ct[329]
    t382 = ct[330]
    t383 = ct[331]
    t384 = ct[332]
    t386 = ct[333]
    t387 = ct[334]
    t388 = ct[335]
    t389 = ct[336]
    t390 = ct[337]
    t391 = ct[338]
    t392 = ct[339]
    t393 = ct[340]
    t394 = ct[341]
    t395 = ct[342]
    t396 = ct[343]
    t397 = ct[344]
    t398 = ct[345]
    t399 = ct[346]
    t4 = ct[347]
    t400 = ct[348]
    t401 = ct[349]
    t402 = ct[350]
    t403 = ct[351]
    t404 = ct[352]
    t405 = ct[353]
    t406 = ct[354]
    t407 = ct[355]
    t408 = ct[356]
    t409 = ct[357]
    t41 = ct[358]
    t410 = ct[359]
    t411 = ct[360]
    t412 = ct[361]
    t413 = ct[362]
    t414 = ct[363]
    t415 = ct[364]
    t416 = ct[365]
    t417 = ct[366]
    t418 = ct[367]
    t419 = ct[368]
    t42 = ct[369]
    t425 = ct[370]
    t428 = ct[371]
    t429 = ct[372]
    t43 = ct[373]
    t430 = ct[374]
    t431 = ct[375]
    t432 = ct[376]
    t433 = ct[377]
    t434 = ct[378]
    t435 = ct[379]
    t436 = ct[380]
    t437 = ct[381]
    t438 = ct[382]
    t439 = ct[383]
    t44 = ct[384]
    t440 = ct[385]
    t441 = ct[386]
    t442 = ct[387]
    t443 = ct[388]
    t444 = ct[389]
    t447 = ct[390]
    t448 = ct[391]
    t449 = ct[392]
    t45 = ct[393]
    t450 = ct[394]
    t451 = ct[395]
    t452 = ct[396]
    t453 = ct[397]
    t455 = ct[398]
    t456 = ct[399]
    t457 = ct[400]
    t458 = ct[401]
    t459 = ct[402]
    t46 = ct[403]
    t460 = ct[404]
    t461 = ct[405]
    t462 = ct[406]
    t463 = ct[407]
    t464 = ct[408]
    t465 = ct[409]
    t466 = ct[410]
    t467 = ct[411]
    t468 = ct[412]

    # Continuation of ct{} translation to Python
    t469 = ct[413]
    t47 = ct[414]
    t470 = ct[415]
    t471 = ct[416]
    t480 = ct[417]
    t482 = ct[418]
    t483 = ct[419]
    t484 = ct[420]
    t486 = ct[421]
    t488 = ct[422]
    t489 = ct[423]
    t49 = ct[424]
    t490 = ct[425]
    t491 = ct[426]
    t498 = ct[427]
    t5 = ct[428]
    t50 = ct[429]
    t503 = ct[430]
    t504 = ct[431]
    t505 = ct[432]
    t51 = ct[433]
    t510 = ct[434]
    t511 = ct[435]
    t512 = ct[436]
    t513 = ct[437]
    t514 = ct[438]
    t516 = ct[439]
    t517 = ct[440]
    t518 = ct[441]
    t519 = ct[442]
    t520 = ct[443]
    t521 = ct[444]
    t522 = ct[445]
    t523 = ct[446]
    t524 = ct[447]
    t525 = ct[448]
    t526 = ct[449]
    t527 = ct[450]
    t528 = ct[451]
    t529 = ct[452]
    t53 = ct[453]
    t530 = ct[454]
    t532 = ct[455]
    t533 = ct[456]
    t534 = ct[457]
    t535 = ct[458]
    t536 = ct[459]
    t537 = ct[460]
    t538 = ct[461]
    t539 = ct[462]
    t54 = ct[463]
    t540 = ct[464]
    t541 = ct[465]
    t543 = ct[466]
    t544 = ct[467]
    t545 = ct[468]
    t549 = ct[469]
    t55 = ct[470]
    t551 = ct[471]
    t553 = ct[472]
    t554 = ct[473]
    t555 = ct[474]
    t557 = ct[475]
    t558 = ct[476]
    t559 = ct[477]
    t56 = ct[478]
    t560 = ct[479]
    t561 = ct[480]
    t562 = ct[481]
    t563 = ct[482]
    t564 = ct[483]
    t565 = ct[484]
    t566 = ct[485]
    t567 = ct[486]
    t568 = ct[487]
    t569 = ct[488]
    t57 = ct[489]
    t570 = ct[490]
    t571 = ct[491]
    t572 = ct[492]
    t573 = ct[493]
    t574 = ct[494]
    t575 = ct[495]
    t576 = ct[496]
    t577 = ct[497]
    t578 = ct[498]
    t579 = ct[499]
    t580 = ct[500]
    t581 = ct[501]
    t582 = ct[502]
    t583 = ct[503]
    t584 = ct[504]
    t585 = ct[505]
    t586 = ct[506]
    t587 = ct[507]
    t589 = ct[508]
    t59 = ct[509]
    t590 = ct[510]
    t591 = ct[511]
    t592 = ct[512]

    t593 = ct[513]
    t594 = ct[514]
    t595 = ct[515]
    t596 = ct[516]
    t597 = ct[517]
    t598 = ct[518]
    t599 = ct[519]
    t6 = ct[520]
    t600 = ct[521]
    t601 = ct[522]
    t602 = ct[523]
    t603 = ct[524]
    t604 = ct[525]
    t605 = ct[526]
    t606 = ct[527]
    t607 = ct[528]
    t608 = ct[529]
    t609 = ct[530]
    t610 = ct[531]
    t611 = ct[532]
    t612 = ct[533]
    t613 = ct[534]
    t614 = ct[535]
    t615 = ct[536]
    t618 = ct[537]
    t620 = ct[538]
    t621 = ct[539]
    t622 = ct[540]
    t623 = ct[541]
    t624 = ct[542]
    t625 = ct[543]
    t626 = ct[544]
    t627 = ct[545]
    t628 = ct[546]
    t629 = ct[547]
    t63 = ct[548]
    t630 = ct[549]
    t631 = ct[550]
    t632 = ct[551]
    t633 = ct[552]
    t634 = ct[553]
    t637 = ct[554]
    t638 = ct[555]
    t639 = ct[556]
    t641 = ct[557]
    t642 = ct[558]
    t643 = ct[559]
    t644 = ct[560]
    t645 = ct[561]
    t646 = ct[562]
    t647 = ct[563]
    t648 = ct[564]
    t649 = ct[565]
    t65 = ct[566]
    t650 = ct[567]
    t651 = ct[568]
    t652 = ct[569]
    t653 = ct[570]
    t654 = ct[571]
    t656 = ct[572]
    t657 = ct[573]
    t658 = ct[574]
    t659 = ct[575]
    t66 = ct[576]
    t660 = ct[577]
    t661 = ct[578]
    t662 = ct[579]
    t663 = ct[580]
    t664 = ct[581]
    t665 = ct[582]
    t666 = ct[583]
    t667 = ct[584]
    t668 = ct[585]
    t669 = ct[586]
    t67 = ct[587]
    t670 = ct[588]
    t671 = ct[589]
    t672 = ct[590]
    t673 = ct[591]
    t674 = ct[592]
    t677 = ct[593]
    t679 = ct[594]
    t68 = ct[595]
    t681 = ct[596]
    t683 = ct[597]
    t684 = ct[598]
    t685 = ct[599]
    t686 = ct[600]
    t689 = ct[601]
    t690 = ct[602]
    t691 = ct[603]
    t692 = ct[604]
    t693 = ct[605]
    t694 = ct[606]
    t695 = ct[607]
    t696 = ct[608]
    t697 = ct[609]
    t698 = ct[610]
    t699 = ct[611]
    t7 = ct[612]

    t70 = ct[613]
    t700 = ct[614]
    t704 = ct[615]
    t706 = ct[616]
    t707 = ct[617]
    t708 = ct[618]
    t709 = ct[619]
    t71 = ct[620]
    t710 = ct[621]
    t711 = ct[622]
    t712 = ct[623]
    t713 = ct[624]
    t714 = ct[625]
    t715 = ct[626]
    t716 = ct[627]
    t717 = ct[628]
    t719 = ct[629]
    t720 = ct[630]
    t721 = ct[631]
    t722 = ct[632]
    t723 = ct[633]
    t724 = ct[634]
    t725 = ct[635]
    t727 = ct[636]
    t728 = ct[637]
    t729 = ct[638]
    t73 = ct[639]
    t730 = ct[640]
    t731 = ct[641]
    t732 = ct[642]
    t733 = ct[643]
    t734 = ct[644]
    t736 = ct[645]
    t74 = ct[646]
    t748 = ct[647]
    t75 = ct[648]
    t755 = ct[649]
    t756 = ct[650]
    t757 = ct[651]
    t758 = ct[652]
    t76 = ct[653]
    t760 = ct[654]
    t761 = ct[655]
    t762 = ct[656]
    t77 = ct[657]
    t771 = ct[658]
    t772 = ct[659]
    t774 = ct[660]
    t78 = ct[661]
    t780 = ct[662]
    t781 = ct[663]
    t782 = ct[664]
    t783 = ct[665]
    t784 = ct[666]
    t785 = ct[667]
    t786 = ct[668]
    t787 = ct[669]
    t789 = ct[670]
    t79 = ct[671]
    t793 = ct[672]
    t794 = ct[673]
    t795 = ct[674]
    t796 = ct[675]
    t8 = ct[676]
    t80 = ct[677]
    t800 = ct[678]
    t801 = ct[679]
    t802 = ct[680]
    t803 = ct[681]
    t804 = ct[682]
    t806 = ct[683]
    t807 = ct[684]
    t808 = ct[685]
    t809 = ct[686]
    t81 = ct[687]
    t811 = ct[688]
    t812 = ct[689]
    t814 = ct[690]
    t82 = ct[691]
    t820 = ct[692]
    t821 = ct[693]
    t822 = ct[694]
    t823 = ct[695]
    t824 = ct[696]
    t825 = ct[697]
    t826 = ct[698]
    t827 = ct[699]
    t828 = ct[700]
    t829 = ct[701]
    t830 = ct[702]
    t831 = ct[703]
    t832 = ct[704]
    t833 = ct[705]
    t834 = ct[706]
    t835 = ct[707]
    t837 = ct[708]
    t838 = ct[709]
    t839 = ct[710]
    t840 = ct[711]
    t841 = ct[712]

    t842 = ct[713]
    t843 = ct[714]
    t844 = ct[715]
    t845 = ct[716]
    t846 = ct[717]
    t847 = ct[718]
    t848 = ct[719]
    t849 = ct[720]
    t85 = ct[721]
    t850 = ct[722]
    t851 = ct[723]
    t853 = ct[724]
    t854 = ct[725]
    t855 = ct[726]
    t856 = ct[727]
    t857 = ct[728]
    t859 = ct[729]
    t86 = ct[730]
    t860 = ct[731]
    t861 = ct[732]
    t862 = ct[733]
    t863 = ct[734]
    t865 = ct[735]
    t866 = ct[736]
    t867 = ct[737]
    t868 = ct[738]
    t869 = ct[739]
    t87 = ct[740]
    t871 = ct[741]
    t874 = ct[742]
    t875 = ct[743]
    t876 = ct[744]
    t877 = ct[745]
    t879 = ct[746]
    t88 = ct[747]
    t880 = ct[748]
    t881 = ct[749]
    t882 = ct[750]
    t883 = ct[751]
    t884 = ct[752]
    t885 = ct[753]
    t889 = ct[754]
    t89 = ct[755]
    t890 = ct[756]
    t892 = ct[757]
    t893 = ct[758]
    t894 = ct[759]
    t895 = ct[760]
    t897 = ct[761]
    t9 = ct[762]
    t90 = ct[763]
    t900 = ct[764]
    t902 = ct[765]
    t903 = ct[766]
    t905 = ct[767]
    t906 = ct[768]
    t907 = ct[769]
    t908 = ct[770]
    t910 = ct[771]
    t911 = ct[772]
    t912 = ct[773]
    t913 = ct[774]
    t915 = ct[775]
    t920 = ct[776]
    t922 = ct[777]
    t923 = ct[778]
    t924 = ct[779]
    t927 = ct[780]
    t928 = ct[781]
    t93 = ct[782]
    t931 = ct[783]
    t932 = ct[784]
    t934 = ct[785]
    t936 = ct[786]
    t937 = ct[787]
    t938 = ct[788]
    t939 = ct[789]
    t94 = ct[790]
    t941 = ct[791]
    t945 = ct[792]
    t946 = ct[793]
    t947 = ct[794]
    t948 = ct[795]
    t95 = ct[796]
    t950 = ct[797]
    t951 = ct[798]
    t952 = ct[799]
    t953 = ct[800]
    t954 = ct[801]
    t955 = ct[802]
    t956 = ct[803]
    t957 = ct[804]
    t958 = ct[805]
    t959 = ct[806]
    t96 = ct[807]
    t960 = ct[808]
    t962 = ct[809]
    t970 = ct[810]
    t971 = ct[811]
    t972 = ct[812]

    t974 = ct[813]
    t976 = ct[814]
    t977 = ct[815]
    t978 = ct[816]
    t979 = ct[817]
    t98 = ct[818]
    t980 = ct[819]
    t981 = ct[820]
    t985 = ct[821]
    t986 = ct[822]
    t987 = ct[823]
    t988 = ct[824]
    t989 = ct[825]
    t99 = ct[826]
    t992 = ct[827]
    w = ct[828]
    t1110 = t8 * t727
    t1111 = t8 * t383
    t1113 = t8 * t391
    t1115 = -t521
    t1116 = t35 + t123
    t1117 = t36 + t124
    t1119 = K_p * t731
    t1120 = F_f_mag * t2 * t18 * t78
    t1121 = K_p * t732
    t1122 = K_p * t733
    t1124 = F_s_mag * t3 * t18 * t78
    t1125 = K_p * t734
    t1126 = t8 * t755
    t1127 = t8 * t756
    t1128 = t8 * t757
    t1129 = t8 * t758
    t1130 = t118 * t126
    t1131 = t118 * t129
    t1136 = t7 * t49 * t75
    t1137 = -t563
    t1138 = t6 * t49 * t53
    t1139 = t7 * t49 * t53
    t1141 = t41 * t126
    t1144 = K_p * t9 * t126
    t1150 = t9 * t49 * t82
    t1151 = t825 ** 2
    t1153 = t49 * t549
    t1155 = t9 * t49 * t75
    t1163 = t7 * t49 * t82
    t1172 = -t647
    t1175 = -t650
    t1176 = t6 * t55 * t126
    t1177 = t98 * t126
    t1180 = -t667
    t1183 = -t672
    t1185 = t49 * t146
    t1188 = t98 * t129
    t1189 = t49 * t247
    t1190 = t6 * t59 * t129
    t1192 = -t692
    t1196 = -t700
    t1197 = t49 * t251
    t1198 = t49 * t252
    t1203 = -t712
    t1205 = -t716
    t1207 = K_p * t825
    t1208 = t6 * t18 * t75 * 3.0
    t1209 = t6 * t18 * t77 * 3.0
    t1210 = -t732
    t1211 = -t733
    t1218 = -2.0 * t8 * t53 * t93
    t1219 = t49 * t312
    t1220 = t46 * t287
    t1228 = -t756
    t1229 = -t757
    t1231 = t9 * t18 * t75
    t1232 = t7 * t18 * t77
    t1235 = -2.0 * K_p * t53 * t95
    t1238 = K_p * t924
    t1240 = K_p * t931
    t1241 = t46 * t362
    t1245 = t8 * t614
    t1251 = t49 * t390
    t1252 = t18 * t53 * t55 * 3.0
    t1254 = t18 * t53 * t59 * 2.0
    t1256 = F_f_mag * t2 * t519
    t1257 = F_s_mag * t3 * t516
    t1258 = K_p * t957
    t1259 = K_p * t958
    t1261 = F_f_mag * t2 * t527
    t1262 = F_s_mag * t3 * t524
    t1263 = K_p * t972
    t1265 = t49 * t417
    t1267 = -2.0 * t8 * t63 * t78
    t1270 = t8 * t683
    t1271 = t8 * t684
    t1275 = t49 * t455
    t1276 = F_s_mag * t45 * t184
    t1278 = -3.0 * t6 * t8 ** 2 * t77
    t1279 = t49 * t560
    t1283 = t9 * t18 * t126
    t1286 = t3 * t719
    t1294 = t49 * t256
    t1297 = t49 * t264

    t1298 = t5 * t719
    t1299 = t8 * t859
    t1300 = t8 * t860
    t1305 = t49 * t658
    t1306 = t49 * t299
    t1308 = -3.0 * t8 ** 2 * t53 * t55
    t1309 = t49 * t307
    t1313 = -2.0 * t8 ** 2 * t53 * t59
    t1314 = t49 * t685
    t1316 = t49 * t328
    t1319 = t49 * t694
    t1320 = t49 * t337
    t1327 = -t1072
    t1330 = t46 * t672
    t1333 = t18 * t59 * t129
    t1339 = -t1098
    t1345 = 2.0 * F_f_mag * t2 * t18 * t98
    t1346 = 2.0 * F_s_mag * t3 * t18 * t98
    t1348 = t49 * t391
    t1352 = -t1273
    t1353 = -t1123
    t1355 = t49 * t755
    t1356 = t49 * t758
    t1357 = t8 * t825
    t1358 = F_f_mag * t55 * t183
    t1359 = F_f_mag * t59 * t183
    t1360 = F_s_mag * t55 * t184
    t1361 = F_s_mag * t57 * t184
    t1368 = -2.0 * F_s_mag * t3 * t8 ** 2 * t98
    t1369 = t49 * t656
    t1372 = F_f_mag * t6 * t183 * 1j
    t1373 = F_f_mag * t9 * t183 * 1j
    t1374 = F_s_mag * t6 * t184 * 1j
    t1375 = F_s_mag * t7 * t184 * 1j
    t1381 = t3 * t1376
    t1382 = t67 + t85 + t103
    t1383 = t68 + t86 + t104
    t1384 = t5 * t1376
    t1388 = t45 + t46 + t57 + t59 + t87
    t1392 = t6 + t7 + t88 + t157 + t159
    t1393 = t89 + t109 + t157 + t160
    t1406 = t88 + t89 + t158 + t159 + t160 + t825
    t277 = -t181
    t445 = -t200
    t446 = -t203
    t487 = -t142
    t492 = -t220
    t508 = -t147
    t509 = -t148
    t515 = -t162
    t531 = K_p * t6 * t114
    t635 = t8 * t557
    t636 = t8 * t559
    t640 = 2.0 * t41 * t162
    t675 = t7 * t147
    t680 = t9 * t151
    t702 = t6 * t255
    t705 = t6 * t263
    t735 = K_p * t280
    t737 = t8 * t301
    t738 = K_p * t281
    t739 = t8 * t302
    t740 = K_p * t282
    t741 = t8 * t303
    t742 = K_p * t289
    t744 = t8 * t313
    t745 = K_p * t290
    t746 = t8 * t314
    t747 = K_p * t291
    t749 = t8 * t325
    t750 = t8 * t326
    t751 = t8 * t327
    t752 = t8 * t334
    t753 = t8 * t335
    t754 = t9 * t333
    t759 = K_p * t342
    t763 = t57 * t167
    t764 = K_p * t354
    t765 = t8 * t376
    t766 = K_p * t355
    t768 = K_p * t365
    t769 = K_p * t366
    t773 = t8 * t384
    t775 = t8 * t386
    t776 = t8 * t388
    t777 = t8 * t389
    t779 = -t465
    t788 = -t480
    t790 = -t482
    t791 = -t483
    t792 = -t484
    t797 = K_p * t397
    t805 = -t498
    t810 = -t503
    t813 = K_p * t405
    t816 = t8 * t418
    t817 = K_p * t407
    t818 = K_p * t409
    t819 = K_p * t410
    t852 = -t587

    t858 = -t263
    t864 = t6 * t18 * t114
    t872 = t74 * t179
    t873 = -t281
    t878 = t9 * t55 * t127
    t886 = -t301
    t887 = -t305
    t888 = -t309
    t891 = -t314
    t896 = -t318
    t898 = t8 * t46 * t144
    t899 = F_s_mag * K_p * t3 * t180
    t904 = t6 * t57 * t130
    t909 = -t336
    t914 = -t629
    t916 = -t344
    t917 = -t630
    t918 = -t632
    t919 = -t350
    t921 = -t354
    t925 = -t639
    t926 = F_f_mag * K_p * t4 * t179
    t930 = -t363
    t933 = -t642
    t935 = -t371
    t940 = -t659
    t942 = -t378
    t943 = -t660
    t944 = -t662
    t949 = -t386
    t961 = F_s_mag * K_p * t3 * t128
    t963 = t8 * t589
    t964 = t8 * t590
    t965 = t8 * t558
    t966 = t8 * t592
    t968 = t8 * t561
    t969 = t8 * t595
    t973 = t79 * t128
    t975 = -t410
    t990 = F_s_mag * K_p * t5 * t128
    t991 = K_p * t596
    t993 = K_p * t597
    t994 = K_p * t598
    t995 = K_p * t601
    t997 = t8 * t610
    t998 = K_p * t602
    t999 = K_p * t603
    t1001 = t8 * t613
    t1004 = t8 * t253
    t1005 = t8 * t620
    t1006 = K_p * t604
    t1008 = K_p * t605
    t1009 = t8 * t621
    t1011 = t8 * t257
    t1012 = K_p * t607
    t1013 = t8 * t622
    t1016 = t8 * t626
    t1017 = K_p * t609
    t1019 = t8 * t627
    t1020 = t8 * t628
    t1022 = K_p * t630
    t1024 = t8 * t646
    t1025 = t8 * t648
    t1027 = t8 * t652
    t1028 = t8 * t654
    t1030 = K_p * t637
    t1033 = t8 * t660
    t1041 = t8 * t308
    t1042 = K_p * t649
    t1045 = t8 * t317
    t1046 = K_p * t657
    t1048 = t8 * t321
    t1053 = t8 * t324
    t1056 = t8 * t329
    t1057 = t8 * t330
    t1058 = t8 * t690
    t1059 = t8 * t331
    t1062 = t8 * t333
    t1065 = t8 * t338
    t1066 = t8 * t339
    t1067 = t8 * t699
    t1068 = t8 * t340
    t1070 = K_p * t666
    t1081 = K_p * t686
    t1086 = t8 * t711
    t1087 = K_p * t689
    t1093 = K_p * t693
    t1096 = K_p * t695
    t1105 = K_p * t708
    t1109 = t8 * t725
    t1112 = t8 * t387
    t1114 = t8 * t392
    t1132 = -t841
    t1134 = -t557
    t1135 = -t842
    t1140 = -t845
    t1142 = t6 * t9 * t110
    t1146 = K_p * t9 * t127
    t1147 = -t848
    t1148 = -t583

    t1154 = -t590
    t1156 = t118 * t127
    t1157 = t110 * t118
    t1159 = K_p * t7 * t127
    t1160 = t41 * t127
    t1161 = t118 * t130
    t1164 = -t609
    t1165 = K_p * t1131
    t1166 = t114 * t118
    t1168 = -t625
    t1169 = -t626
    t1170 = t49 * t559
    t1171 = -t936
    t1173 = -t648
    t1174 = t75 * t180
    t1179 = -t666
    t1181 = -t946
    t1182 = -t671
    t1186 = K_p * t1141
    t1191 = t94 * t130
    t1193 = -t695
    t1194 = -t955
    t1200 = t8 * t842
    t1201 = t8 * t843
    t1202 = -t711
    t1204 = -t971
    t1214 = t49 * t302
    t1215 = t49 * t303
    t1216 = -2.0 * t6 * t311
    t1222 = t8 * t591
    t1223 = t8 * t594
    t1224 = -2.0 * t6 * t329
    t1225 = t49 * t327
    t1226 = t49 * t334
    t1227 = t49 * t335
    t1234 = K_p * t913
    t1237 = K_p * t922
    t1239 = -2.0 * K_p * t41 * t162
    t1244 = t8 * t612
    t1246 = t8 * t615
    t1249 = t49 * t384
    t1250 = t49 * t389
    t1266 = t49 * t418
    t1269 = -t822
    t1274 = -t829
    t1277 = t49 * t589
    t1280 = t49 * t561
    t1281 = t49 * t595
    t1282 = -t1209
    t1284 = t9 * t18 * t127
    t1291 = t49 * t620
    t1293 = t49 * t621
    t1295 = t49 * t257
    t1303 = t49 * t646
    t1304 = t49 * t654
    t1307 = -t1039
    t1310 = t49 * t308
    t1312 = K_p * t1176
    t1315 = t49 * t324
    t1317 = t49 * t330
    t1318 = t49 * t333
    t1321 = t49 * t339
    t1322 = t49 * t699
    t1325 = -t1252
    t1328 = -t1073
    t1329 = -t1254
    t1331 = K_p * t1188
    t1332 = t7 * t18 * t514
    t1334 = F_f_mag * t4 * t18 * t179
    t1336 = -t1095
    t1340 = -t1100
    t1342 = -t1104
    t1343 = t8 * t936
    t1344 = t8 * t937
    t1349 = t49 * t392
    t1350 = -t1121
    t1351 = -t1122
    t1354 = F_s_mag * t3 * t18 * t128
    t1364 = t49 * t594
    t1366 = t7 * t18 * t127
    t1370 = -t1346
    t1371 = -t1262
    t1377 = -t1358
    t1378 = -t1359
    t1379 = -t1374
    t1380 = -t1375
    t1385 = t43 + t65 + t192
    t1386 = t44 + t66 + t193
    t1387 = -t1384
    t1389 = K_p * t1388
    t1390 = t1388 ** 2
    t1391 = t8 * t1388
    t1394 = t49 * t1388
    t1395 = t31 + t32 + t120 + t177 + t178
    t1397 = t41 + t42 + t121 + t179 + t180
    t1399 = t1298 + t1381
    t1407 = t71 + t90 + t95 + t96 + t98 + t128 + t238

    t1423 = -1.0 / (
            t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239
            - t240 - t241 + t46 * t57 + m_s * t50 * w
    )

    t1424 = 1.0 / (
            t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239
            - t240 - t241 + t46 * t57 + m_s * t50 * w
    ) ** 2

    t1455 = -1.0 / (
            (t618 * t1406) / (
            t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239
            - t240 - t241 + t46 * t57 + m_s * t50 * w
    ) + 1.0
    )

    t929 = -t640
    t1032 = K_p * t640
    t1187 = K_p * t1142
    t1199 = -t705
    t1212 = K_p * t872
    t1213 = -t738
    t1221 = K_p * t878
    t1236 = -t764
    t1264 = K_p * t973
    t1268 = -t819
    t1287 = K_p * t1157
    t1288 = K_p * t1160
    t1292 = K_p * t1161
    t1296 = -t1017
    t1311 = K_p * t1174
    t1324 = -t1070
    t1337 = -t1096
    t1396 = t1395 ** 2
    t1398 = t1397 ** 2
    t1401 = t1286 + t1387
    t1402 = t1151 + t1390
    t1403 = t1357 + t1389
    t1404 = t1207 + t1394
    t1408 = t1407 ** 2
    t1411 = t53 + t81 + t82 + t110 + t129 + t130 + t143 + t144 + t486 + t487
    t1413 = t63 + t74 + t75 + t114 + t126 + t127 + t163 + t164 + t514 + t515
    t1422 = t139 + t140 + t240 + t241 + t555 + t1407
    t1425 = t122 + t232 + t242 + t243 + t246 + t277 + t840 + t1397
    t1447 = t618 * t1406 * t1423
    t1452 = t869 + t1276 + t1360 + t1361 + t1372 + t1373 + t1377 + t1378 + t1379 + t1380
    t1516 = (
            t111 + t112 + t113 + t115 + t116 + t117 + t119 + t125 + t186 + t188 + t189 +
            t191 + t194 + t195 + t196 + t197 + t198 + t199 + t201 + t202 + t205 + t221 +
            t222 + t223 + t224 + t225 + t226 + t227 + t296 + t403 + t425 + t428 + t429 +
            t430 + t431 + t432 + t433 + t434 + t435 + t436 + t439 + t440 + t441 + t442 +
            t443 + t444 + t445 + t446 + t447 + t448 + t449 + t450 + t451 + t452 + t453 +
            t456 + t457 + t458 + t459 + t460 + t461 + t462 + t463 + t464 + t466 + t467 +
            t468 + t469 + t470 + t471 + t488 + t489 + t490 + t491 + t492 + t504 + t505 +
            t516 + t517 + t518 + t519 + t524 + t525 + t526 + t527 + t534 + t535 + t536 +
            t537 + t538 + t539 + t540 + t541 + t543 + t544 + t545 + t551 + t728 + t729 +
            t730 + t761 + t772 + t779 + t780 + t781 + t782 + t783 + t784 + t785 + t786 +
            t787 + t788 + t789 + t790 + t791 + t792 + t793 + t800 + t801 + t802 + t803 +
            t804 + t805 + t806 + t807 + t808 + t809 + t810 + t811 + t812 + t823 + t824 +
            t826 + t831 + t832 + t833 + t834 + t839 + t1132
    )
    t1527 = (
            t101 + t135 + t136 + t145 + t146 + t150 + t151 + t155 + t156 + t187 + t230 +
            t231 + t247 + t249 + t250 + t251 + t252 + t255 + t256 + t257 + t264 + t298 +
            t299 + t302 + t303 + t304 + t306 + t307 + t308 + t311 + t312 + t313 + t316 +
            t319 + t324 + t327 + t328 + t330 + t333 + t334 + t335 + t337 + t339 + t379 +
            t380 + t381 + t382 + t384 + t389 + t390 + t391 + t392 + t414 + t415 + t416 +
            t417 + t418 + t419 + t455 + t508 + t509 + t510 + t511 + t512 + t513 + t549 +
            t559 + t560 + t561 + t586 + t589 + t594 + t595 + t611 + t612 + t613 + t615 +
            t620 + t621 + t623 + t627 + t646 + t654 + t656 + t658 + t661 + t663 + t664 +
            t681 + t684 + t685 + t694 + t699 + t713 + t714 + t715 + t717 + t724 + t725 +
            t755 + t758 + t827 + t828 + t843 + t849 + t850 + t851 + t852 + t853 + t854 +
            t855 + t856 + t857 + t858 + t860 + t883 + t884 + t885 + t886 + t887 + t888 +
            t889 + t890 + t891 + t892 + t893 + t894 + t895 + t896 + t897 + t898 + t899 +
            t900 + t902 + t903 + t904 + t905 + t906 + t907 + t908 + t909 + t910 + t911 +
            t937 + t938 + t939 + t940 + t941 + t942 + t943 + t944 + t947 + t948 + t949 +
            t950 + t951 + t952 + t953 + t970 + t985 + t986 + t987 + t988 + t989 + t990 +
            t1036 + t1130 + t1133 + t1134 + t1135 + t1136 + t1152 + t1154 + t1155 + t1156 +
            t1166 + t1167 + t1168 + t1169 + t1171 + t1173 + t1175 + t1177 + t1178 + t1190 +
            t1191 + t1192 + t1202 + t1203 + t1204 + t1205 + t1206 + t1228 + t1229
    )
    t1400 = t25 * t1396
    t1405 = 1.0 / t1402
    t1416 = t1398 + t1408
    t1426 = 1.0 / t1425
    t1451 = t1447 - 1.0
    t1488 = (
            (t618 * t1393 * t1423 * t1452 * w) /
            (((t618 * t1406) / (
                        t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239 - t240 - t241 + t46 * t57 + m_s * t50 * w) + 1.0) *
             (t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239 - t240 - t241 + t46 * t57 + m_s * t50 * w))
    )
    t1517 = 1.0 / t1516
    t1528 = (
            t99 + t131 + t132 + t167 + t168 + t169 + t172 + t173 + t174 + t185 + t233 +
            t234 + t279 + t282 + t283 + t285 + t288 + t289 + t290 + t292 + t294 + t345 +
            t346 + t347 + t348 + t349 + t352 + t353 + t357 + t361 + t362 + t366 + t367 +
            t369 + t370 + t372 + t373 + t374 + t393 + t394 + t395 + t396 + t397 + t398 +
            t404 + t406 + t407 + t408 + t409 + t438 + t528 + t529 + t530 + t531 + t532 +
            t533 + t554 + t562 + t564 + t565 + t566 + t567 + t570 + t571 + t572 + t579 +
            t580 + t581 + t582 + t596 + t597 + t599 + t602 + t605 + t606 + t607 + t631 +
            t633 + t634 + t637 + t641 + t643 + t644 + t645 + t653 + t657 + t668 + t669 +
            t670 + t673 + t689 + t691 + t693 + t696 + t707 + t708 + t721 + t722 + t731 +
            t734 + t837 + t838 + t844 + t847 + t867 + t871 + t872 + t873 + t874 + t875 +
            t876 + t877 + t878 + t879 + t880 + t881 + t912 + t913 + t914 + t915 + t916 +
            t917 + t918 + t919 + t920 + t921 + t922 + t923 + t924 + t925 + t926 + t927 +
            t928 + t929 + t930 + t931 + t932 + t933 + t934 + t935 + t945 + t954 + t956 +
            t957 + t958 + t959 + t960 + t961 + t972 + t973 + t974 + t975 + t976 + t977 +
            t978 + t1021 + t1131 + t1137 + t1138 + t1139 + t1140 + t1141 + t1142 + t1143 +
            t1144 + t1145 + t1146 + t1147 + t1148 + t1149 + t1150 + t1157 + t1158 + t1159 +
            t1160 + t1161 + t1162 + t1163 + t1164 + t1172 + t1174 + t1176 + t1179 + t1180 +
            t1181 + t1183 + t1184 + t1188 + t1193 + t1194 + t1195 + t1196 + t1210 + t1211
        )


    et1 = (
        t212 + t213 + t245 + t266 + t267 + t268 + t272 + t273 + t274 + t520 + t593 + t624 +
        t635 + t665 + t675 + t677 + t679 + t697 + t702 + t704 + t706 + t736 + t737 + t740 +
        t742 + t745 + t746 + t748 + t749 + t750 + t754 + t760 + t762 + t763 + t765 + t769 +
        t771 + t774 + t775 + t776 + t794 + t795 + t796 + t797 + t814 + t817 + t818 + t820 +
        t821 + t830 + t835 + t846 + t861 + t862 + t863 + t864 + t865 + t866 + t962 + t964 +
        t965 + t966 + t979 + t980 + t981 + t991 + t992 + t993 + t997 + t998 + t1002 + t1004 +
        t1007 + t1008 + t1012 + t1013 + t1014 + t1015 + t1016 + t1020 + t1023 + t1025 + t1026 +
        t1027 + t1030 + t1033 + t1035 + t1037 + t1044 + t1045 + t1046 + t1047 + t1048 + t1050 +
        t1052 + t1055 + t1056 + t1058 + t1059 + t1060 + t1061 + t1065 + t1068 + t1074 + t1075 +
        t1076 + t1077 + t1078 + t1079 + t1082 + t1085 + t1086 + t1087 + t1089 + t1090 + t1091 +
        t1093 + t1094 + t1099 + t1102 + t1103 + t1105 + t1108 + t1110 + t1111 + t1112 + t1115 +
        t1118 + t1119 + t1120 + t1125 + t1127 + t1128 + t1153 + t1165 + t1170 + t1182 + t1185 +
        t1186 + t1187 + t1189 + t1197 + t1198 + t1199 + t1200 + t1208 + t1212 + t1213 + t1214 +
        t1215 + t1216 + t1217 + t1218 + t1219 + t1220 + t1221 + t1222 + t1224 + t1225 + t1226 +
        t1227 + t1230 + t1231 + t1234 + t1235 + t1236 + t1237 + t1238 + t1239 + t1240 + t1241 +
        t1242 + t1243 + t1245 + t1247 + t1248 + t1249 + t1250 + t1251 + t1253 + t1255 + t1256 +
        t1258 + t1259 + t1260 + t1261 + t1263 + t1264 + t1265 + t1266 + t1267 + t1268 + t1269 +
        t1270 + t1272 + t1274 + t1275 + t1277 + t1278 + t1279
    )

    et2 = (
            t1280 + t1281 + t1282 + t1283 + t1284 + t1285 + t1287 + t1288 + t1289 + t1290 + t1291 +
            t1292 + t1293 + t1294 + t1295 + t1296 + t1297 + t1299 + t1301 + t1303 + t1304 + t1305 +
            t1306 + t1307 + t1308 + t1309 + t1310 + t1311 + t1312 + t1313 + t1314 + t1315 + t1316 +
            t1317 + t1318 + t1319 + t1320 + t1321 + t1322 + t1323 + t1324 + t1325 + t1326 + t1327 +
            t1328 + t1329 + t1330 + t1331 + t1332 + t1333 + t1334 + t1335 + t1336 + t1337 + t1338 +
            t1339 + t1340 + t1341 + t1342 + t1343 + t1345 + t1347 + t1348 + t1349 + t1350 + t1351 +
            t1352 + t1353 + t1354 + t1355 + t1356 + t1363 + t1364 + t1365 + t1366 + t1367 + t1368 +
            t1369 + t1370 + t1371
    )

    t1530 = et1 + et2
    t1409 = t109 * t825 * t1405
    t1410 = t825 * t1362 * t1405
    t1412 = t109 * t1388 * t1405
    t1415 = t1362 * t1388 * t1405
    t1417 = t1400 + t1408
    t1418 = 1.0 / t1416
    t1420 = t825 * t1397 * t1405
    t1427 = t1388 * t1397 * t1405
    t1428 = t825 * t1405 * t1407
    t1430 = t1388 * t1405 * t1407
    t1489 = t1488.image
    t1490 = t1488.real
    t1492 = (
            (t618 * t1392 * t1426 * t1452 * w) /
            (((t618 * t1406) / (
                        t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239 - t240 - t241 + t46 * t57 + m_s * t50 * w) + 1.0) *
             (t33 + t78 - t90 - t98 + t107 + t108 + t118 + t239 - t240 - t241 + t46 * t57 + m_s * t50 * w))
    )
    t1518 = t1517 ** 2
    t1414 = -t1412
    t1419 = t1418 ** 2
    t1421 = 1.0 / t1417
    t1429 = -t1428
    t1432 = t825 * t1397 * t1418
    t1438 = t1388 * t1397 * t1418
    t1439 = t825 * t1407 * t1418
    t1440 = t1409 + t1415
    t1456 = t1388 * t1407 * t1418
    t1457 = t1397 * t1403 * t1418
    t1458 = t1397 * t1404 * t1418
    t1460 = t1403 * t1407 * t1418
    t1461 = t1404 * t1407 * t1418
    t1462 = t1397 * t1411 * t1418
    t1463 = t1397 * t1413 * t1418
    t1465 = t1420 + t1430
    t1467 = t1407 * t1411 * t1418
    t1468 = t1407 * t1413 * t1418
    t1491 = t47 * t1490
    t1493 = t1492.imag
    t1494 = t1492.real
    t1431 = t1398 * t1419
    t1433 = F_f_mag * t25 * t1116 * t1395 * t1421
    t1434 = F_s_mag * t25 * t1117 * t1395 * t1421
    t1435 = t1408 * t1419
    t1436 = F_f_mag * t1386 * t1395 * t1421 * w
    t1437 = F_s_mag * t1385 * t1395 * t1421 * w
    t1441 = -t1439
    t1442 = F_f_mag * t522 * t1407 * t1421 * w
    t1443 = F_s_mag * t523 * t1407 * t1421 * w
    t1444 = t1410 + t1414
    t1445 = F_f_mag * t2 * t1440
    t1446 = F_f_mag * t4 * t1440
    t1453 = F_f_mag * t1383 * t1407 * t1421
    t1454 = F_s_mag * t1382 * t1407 * t1421
    t1459 = -t1457
    t1464 = -t1463
    t1466 = t1427 + t1429
    t1470 = t1432 + t1456
    t1477 = t1397 * t1418 * t1465
    t1480 = t1407 * t1418 * t1465
    t1481 = t1458 + t1460 + 1.0
    t1485 = (t1457 - t1461) ** 2

    t1486 = t1462 + t1468
    t1495 = t47 * t1493
    t1448 = F_f_mag * t2 * t1444
    t1449 = F_f_mag * t4 * t1444
    t1469 = t1431 + t1435
    t1471 = K_p * t1470
    t1472 = t1438 + t1441
    t1473 = t8 * t1470
    t1478 = t1397 * t1418 * t1466
    t1479 = -t1477
    t1482 = t1407 * t1418 * t1466
    t1483 = t1459 + t1461
    t1484 = t1481 ** 2
    t1487 = t1464 + t1467
    t1450 = -t1449
    t1474 = K_p * t1472



    # Example usage, assuming ft_3 is a defined Python function
    mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s = ft_3([
        F_f_mag, F_s_mag, K_p, m_c, m_f, m_s, t10, t100, t1116, t1117, t126, t127, t128, t129, t130, t132, t133, t134,
        t1382, t1383, t1385, t1386, t1395, t1399, t1401, t1402, t1407, t141, t1421, t143, t1433, t1434, t1436, t1437,
        t144, t1442, t1443, t1445, t1446, t1448, t1449, t1450, t1453, t1454, t1457, t1461, t1463, t1467, t1469, t1471,
        t1472, t1473, t1474, t1477, t1478, t1479, t1480, t1481, t1482, t1484, t1485, t1486, t1489, t1491, t1494, t1495,
        t1517, t1518, t1527, t1530, t163, t164, t165, t166, t167, t168, t170, t171, t174, t175, t176, t180, t19, t2,
        t20,
        t233, t235, t236, t238, t24, t25, t26, t27, t278, t280, t281, t282, t284, t286, t287, t291, t293, t295, t3, t33,
        t341, t342, t343, t344, t350, t351, t353, t354, t355, t356, t358, t359, t360, t362, t363, t364, t365, t366,
        t368,
        t370, t371, t395, t397, t399, t4, t400, t401, t402, t405, t406, t410, t411, t412, t413, t42, t437, t438, t45,
        t46,
        t47, t49, t5, t50, t51, t522, t523, t53, t54, t55, t553, t56, t562, t563, t564, t566, t567, t568, t569, t57,
        t570,
        t573, t574, t575, t576, t577, t578, t583, t584, t585, t59, t597, t598, t599, t6, t600, t601, t603, t604, t608,
        t609, t629, t63, t630, t631, t632, t633, t634, t638, t639, t640, t641, t642, t643, t644, t645, t647, t649, t651,
        t666, t667, t669, t672, t674, t686, t691, t693, t695, t698, t7, t70, t700, t707, t708, t709, t71, t710, t720,
        t722, t723, t73, t731, t732, t733, t734, t74, t75, t76, t77, t78, t79, t8, t80, t81, t82, t844, t845, t847,
        t848,
        t868, t882, t9, t90, t93, t94, t945, t946, t95, t954, t955, t98, w
    ])

    return mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s


def ft_3(ct):
    # Initialize output variables
    mag_U = 0
    phase_U = 0
    real_P = 0
    imag_P = 0
    mag_X_u = 0
    phase_X_u = 0
    mag_X_f = 0
    phase_X_f = 0
    mag_X_s = 0
    phase_X_s = 0

    # Extract parameters from ct list (Python is 0-based indexing)
    F_f_mag = ct[0]
    F_s_mag = ct[1]
    K_p = ct[2]
    m_c = ct[3]
    m_f = ct[4]
    m_s = ct[5]

    t10 = ct[6]
    t100 = ct[7]
    t1116 = ct[8]
    t1117 = ct[9]
    t126 = ct[10]
    t127 = ct[11]
    t128 = ct[12]
    t129 = ct[13]
    t130 = ct[14]
    t132 = ct[15]
    t133 = ct[16]
    t134 = ct[17]
    t1382 = ct[18]
    t1383 = ct[19]
    t1385 = ct[20]
    t1386 = ct[21]
    t1395 = ct[22]
    t1399 = ct[23]
    t1401 = ct[24]
    t1402 = ct[25]
    t1407 = ct[26]
    t141 = ct[27]
    t1421 = ct[28]
    t143 = ct[29]
    t1433 = ct[30]
    t1434 = ct[31]
    t1436 = ct[32]
    t1437 = ct[33]
    t144 = ct[34]
    t1442 = ct[35]
    t1443 = ct[36]
    t1445 = ct[37]
    t1446 = ct[38]
    t1448 = ct[39]
    t1449 = ct[40]
    t1450 = ct[41]
    t1453 = ct[42]
    t1454 = ct[43]
    t1457 = ct[44]
    t1461 = ct[45]
    t1463 = ct[46]
    t1467 = ct[47]
    t1469 = ct[48]
    t1471 = ct[49]
    t1472 = ct[50]
    t1473 = ct[51]
    t1474 = ct[52]
    t1477 = ct[53]
    t1478 = ct[54]
    t1479 = ct[55]
    t1480 = ct[56]
    t1481 = ct[57]
    t1482 = ct[58]
    t1484 = ct[59]
    t1485 = ct[60]
    t1486 = ct[61]
    t1489 = ct[62]
    t1491 = ct[63]
    t1494 = ct[64]
    t1495 = ct[65]
    t1517 = ct[66]
    t1518 = ct[67]
    t1527 = ct[68]
    t1530 = ct[69]
    t163 = ct[70]

    t164 = ct[71]
    t165 = ct[72]
    t166 = ct[73]
    t167 = ct[74]
    t168 = ct[75]
    t170 = ct[76]
    t171 = ct[77]
    t174 = ct[78]
    t175 = ct[79]
    t176 = ct[80]
    t180 = ct[81]
    t19 = ct[82]
    t2 = ct[83]
    t20 = ct[84]
    t233 = ct[85]
    t235 = ct[86]
    t236 = ct[87]
    t238 = ct[88]
    t24 = ct[89]
    t25 = ct[90]
    t26 = ct[91]
    t27 = ct[92]
    t278 = ct[93]
    t280 = ct[94]
    t281 = ct[95]
    t282 = ct[96]
    t284 = ct[97]
    t286 = ct[98]
    t287 = ct[99]
    t291 = ct[100]
    t293 = ct[101]
    t295 = ct[102]
    t3 = ct[103]
    t33 = ct[104]
    t341 = ct[105]
    t342 = ct[106]
    t343 = ct[107]
    t344 = ct[108]
    t350 = ct[109]
    t351 = ct[110]
    t353 = ct[111]
    t354 = ct[112]
    t355 = ct[113]
    t356 = ct[114]
    t358 = ct[115]
    t359 = ct[116]
    t360 = ct[117]
    t362 = ct[118]
    t363 = ct[119]
    t364 = ct[120]
    t365 = ct[121]
    t366 = ct[122]
    t368 = ct[123]
    t370 = ct[124]
    t371 = ct[125]
    t395 = ct[126]
    t397 = ct[127]
    t399 = ct[128]
    t4 = ct[129]
    t400 = ct[130]
    t401 = ct[131]
    t402 = ct[132]
    t405 = ct[133]
    t406 = ct[134]
    t410 = ct[135]
    t411 = ct[136]
    t412 = ct[137]
    t413 = ct[138]
    t42 = ct[139]
    t437 = ct[140]
    t438 = ct[141]
    t45 = ct[142]
    t46 = ct[143]
    t47 = ct[144]
    t49 = ct[145]
    t5 = ct[146]
    t50 = ct[147]
    t51 = ct[148]
    t522 = ct[149]
    t523 = ct[150]
    t53 = ct[151]
    t54 = ct[152]
    t55 = ct[153]
    t553 = ct[154]
    t56 = ct[155]
    t562 = ct[156]
    t563 = ct[157]
    t564 = ct[158]
    t566 = ct[159]
    t567 = ct[160]
    t568 = ct[161]
    t569 = ct[162]
    t57 = ct[163]
    t570 = ct[164]
    t573 = ct[165]
    t574 = ct[166]
    t575 = ct[167]
    t576 = ct[168]
    t577 = ct[169]
    t578 = ct[170]

    t583 = ct[171]
    t584 = ct[172]
    t585 = ct[173]
    t59 = ct[174]
    t597 = ct[175]
    t598 = ct[176]
    t599 = ct[177]
    t6 = ct[178]
    t600 = ct[179]
    t601 = ct[180]
    t603 = ct[181]
    t604 = ct[182]
    t608 = ct[183]
    t609 = ct[184]
    t629 = ct[185]
    t63 = ct[186]
    t630 = ct[187]
    t631 = ct[188]
    t632 = ct[189]
    t633 = ct[190]
    t634 = ct[191]
    t638 = ct[192]
    t639 = ct[193]
    t640 = ct[194]
    t641 = ct[195]
    t642 = ct[196]
    t643 = ct[197]
    t644 = ct[198]
    t645 = ct[199]
    t647 = ct[200]
    t649 = ct[201]
    t651 = ct[202]
    t666 = ct[203]
    t667 = ct[204]
    t669 = ct[205]
    t672 = ct[206]
    t674 = ct[207]
    t686 = ct[208]
    t691 = ct[209]
    t693 = ct[210]
    t695 = ct[211]
    t698 = ct[212]
    t7 = ct[213]
    t70 = ct[214]
    t700 = ct[215]
    t707 = ct[216]
    t708 = ct[217]
    t709 = ct[218]
    t71 = ct[219]
    t710 = ct[220]
    t720 = ct[221]
    t722 = ct[222]
    t723 = ct[223]
    t73 = ct[224]
    t731 = ct[225]
    t732 = ct[226]
    t733 = ct[227]
    t734 = ct[228]
    t74 = ct[229]
    t75 = ct[230]
    t76 = ct[231]
    t77 = ct[232]
    t78 = ct[233]
    t79 = ct[234]
    t8 = ct[235]
    t80 = ct[236]
    t81 = ct[237]
    t82 = ct[238]
    t844 = ct[239]
    t845 = ct[240]
    t847 = ct[241]
    t848 = ct[242]
    t868 = ct[243]
    t882 = ct[244]
    t9 = ct[245]
    t90 = ct[246]
    t93 = ct[247]
    t94 = ct[248]
    t945 = ct[249]
    t946 = ct[250]
    t95 = ct[251]
    t954 = ct[252]
    t955 = ct[253]
    t98 = ct[254]
    w = ct[255]
    t1475 = t8 * t1472
    t1476 = t49 * t1472
    t1500 = t1478 + t1480
    t1501 = t1479 + t1482
    t1502 = t1484 + t1485
    t1506 = -F_s_mag * t47 * t1399 * (t1477 - t1482)
    t1508 = -F_s_mag * t47 * t1401 * (t1477 - t1482)
    t1496 = t1473 + t1474
    t1498 = t1471 + t1476 + 1.0
    t1503 = 1.0 / t1502
    t1504 = F_s_mag * t47 * t1399 * t1500
    t1505 = F_s_mag * t47 * t1401 * t1500
    t1497 = t1496 ** 2
    t1499 = t1498 ** 2
    t1507 = -t1504

    t1509 = K_p * t1481 * t1503
    t1510 = -K_p * t1503 * (t1457 - t1461)
    t1511 = t8 * t1481 * t1503
    t1512 = t49 * t1503 * (t1457 - t1461)
    t1513 = t8 * t1503 * (t1457 - t1461)
    t1519 = t1446 + t1448 + t1505 + t1506
    t1522 = (-t1445 + t1449 + t1504 + F_s_mag * t47 * t1401 * (t1477 - t1482)) ** 2
    t1514 = t1497 + t1499
    t1520 = t1519 ** 2
    t1521 = t1445 + t1450 + t1507 + t1508
    t1525 = t1510 + t1511
    t1526 = t1509 + t1513
    mag_U = np.sqrt((t1486 * t1526 - t1525 * (t1463 - t1467)) ** 2 + t1518 * t1530 ** 2)

    #phase_U
    phase_U = np.arctan2(t1486 * t1526 - t1525 * (t1463 - t1467), -t1517 * t1530)

    #real_P
    # Reciprocal of square of t1514
    t1515 = 1.0 / (t1514 ** 2)

    # Sum of squares
    t1529 = t1520 + t1522

    # Weighted terms using reciprocal
    t1523 = t1497 * t1515
    t1524 = t1499 * t1515
    t1531 = t1523 + t1524

    # Final computation for real_P
    real_P = (t8 * t1402 * t1469 * t1529 * t1531 * w) / 2.0

    # image_P
    imag_P = K_p * t1402 * t1469 * t1529 * t1531 * w * (-0.5)

    # mag_X_u
    # et3 computation
    et3 = (
            t100 - t132 + t133 + t134 + t165 + t166 - t167 - t168 + t170 + t171
            - t174 + t175 + t176 - t233 + t235 + t236 + t278 + t280 + t281 - t282
            + t284 + t286 + t287 + t291 + t293 + t295 + t341 + t342 + t343 + t344
            + t350 + t351 - t353 + t354 + t355 + t356 + t358 + t359 + t360 - t362
            + t363 + t364 + t365 - t366 + t368 - t370 + t371 - t395 - t397 + t399
            + t400 + t401 + t402 + t405 - t406 + t410 + t411 + t412 + t413 + t437
            - t438 + t553 - t562 + t563 - t564 - t566 - t567 + t568 + t569 - t570
            + t573 + t574 + t575 + t576 + t577 + t578 + t583 + t584 + t585 - t597
            + t598 - t599 + t600 + t601 + t603 + t604 + t608 + t609 + t629 + t630
            - t631 + t632 - t633 - t634 + t638 + t639 + t640 - t641 + t642 - t643
            - t644 - t645 + t647 + t649 + t651 + t666 + t667 - t669 + t672 + t674
            + t686
    )

    # et4 computation
    et4 = (
            -t691 - t693 + t695 + t698 + t700 - t707 - t708 + t709 + t710 + t720
            - t722 + t723 - t731 + t732 + t733 - t734 - t844 + t845 - t847 + t848
            + t868 + t882 - t945 + t946 - t954 + t955
            + t71 * t81 + t81 * t128 + t90 * t130 + t93 * t127 + t94 * t126
            + t98 * t130 + t53 * t238 + t81 * t238 + t82 * t238
            - K_p * t6 * t74 * 2.0 + K_p * t46 * t81 + K_p * t55 * t130
            + K_p * t57 * t129 + K_p * t57 * t130 + K_p * t46 * t143
            + K_p * t46 * t144 - t6 * t8 * t79 * 2.0 + t6 * t49 * t54
            + t9 * t49 * t54 - t6 * t55 * t63 * 2.0 + t8 * t45 * t74
            + t7 * t49 * t79 - t6 * t55 * t75 * 2.0 + t7 * t49 * t80
            + t9 * t49 * t80 + t6 * t7 * t130 + t49 * t55 * t77
            + t49 * t59 * t74 + t49 * t59 * t76 + t49 * t59 * t77
            + t6 * t57 * t126 + t6 * t59 * t127 + t7 * t49 * t141
            + t8 * t46 * t163 + t8 * t46 * t164
    )

    # et5 computation
    et5 = (
            F_f_mag * t2 * t19 * t45 + F_f_mag * t4 * t19 * t49 + F_f_mag * t4 * t33 * t49
            + F_f_mag * t2 * t42 * t49 + F_f_mag * t4 * t49 * t78 + F_f_mag * t2 * t49 * t93
            + F_s_mag * t10 * t46 * t70 + F_s_mag * t9 * t50 * t73 + F_s_mag * t5 * t49 * t95
            - m_c * m_f * t27 * t81 * 2.0 + m_c * m_f * t27 * t130
            + F_f_mag * K_p * t2 * t71 + F_f_mag * K_p * t2 * t128
            + F_s_mag * K_p * t5 * t180 + F_f_mag * m_s * t4 * t50 * t51
            + F_f_mag * m_s * t2 * t50 * t56 + F_s_mag * m_s * t3 * t10 * t50
            + F_f_mag * t2 * t24 * t26 * t50 + F_f_mag * t4 * t24 * t27 * t49
            + F_s_mag * t3 * t20 * t26 * t50 + F_s_mag * K_p * m_s * t3 * t50 * w
            + F_f_mag * m_c * m_f * t4 * t27 * t49
    )

    # Final magnitude of displacement response at u
    mag_X_u = np.sqrt(t1518 * t1527 ** 2 + t1518 * (et3 + et4 + et5) ** 2)

    # phase_X_u
    # et6 (same as et3)
    et6 = (
            t100 - t132 + t133 + t134 + t165 + t166 - t167 - t168 + t170 + t171
            - t174 + t175 + t176 - t233 + t235 + t236 + t278 + t280 + t281 - t282
            + t284 + t286 + t287 + t291 + t293 + t295 + t341 + t342 + t343 + t344
            + t350 + t351 - t353 + t354 + t355 + t356 + t358 + t359 + t360 - t362
            + t363 + t364 + t365 - t366 + t368 - t370 + t371 - t395 - t397 + t399
            + t400 + t401 + t402 + t405 - t406 + t410 + t411 + t412 + t413 + t437
            - t438 + t553 - t562 + t563 - t564 - t566 - t567 + t568 + t569 - t570
            + t573 + t574 + t575 + t576 + t577 + t578 + t583 + t584 + t585 - t597
            + t598 - t599 + t600 + t601 + t603 + t604 + t608 + t609 + t629 + t630
            - t631 + t632 - t633 - t634 + t638 + t639 + t640 - t641 + t642 - t643
            - t644 - t645 + t647 + t649 + t651 + t666 + t667 - t669 + t672 + t674
            + t686
    )

    # et7 (same as et4)
    et7 = (
            -t691 - t693 + t695 + t698 + t700 - t707 - t708 + t709 + t710 + t720
            - t722 + t723 - t731 + t732 + t733 - t734 - t844 + t845 - t847 + t848
            + t868 + t882 - t945 + t946 - t954 + t955
            + t71 * t81 + t81 * t128 + t90 * t130 + t93 * t127 + t94 * t126
            + t98 * t130 + t53 * t238 + t81 * t238 + t82 * t238
            - K_p * t6 * t74 * 2.0 + K_p * t46 * t81 + K_p * t55 * t130
            + K_p * t57 * t129 + K_p * t57 * t130 + K_p * t46 * t143
            + K_p * t46 * t144 - t6 * t8 * t79 * 2.0 + t6 * t49 * t54
            + t9 * t49 * t54 - t6 * t55 * t63 * 2.0 + t8 * t45 * t74
            + t7 * t49 * t79 - t6 * t55 * t75 * 2.0 + t7 * t49 * t80
            + t9 * t49 * t80 + t6 * t7 * t130 + t49 * t55 * t77
            + t49 * t59 * t74 + t49 * t59 * t76 + t49 * t59 * t77
            + t6 * t57 * t126 + t6 * t59 * t127 + t7 * t49 * t141
            + t8 * t46 * t163 + t8 * t46 * t164
    )

    # et8 (same as et5)
    et8 = (
            F_f_mag * t2 * t19 * t45 + F_f_mag * t4 * t19 * t49 + F_f_mag * t4 * t33 * t49
            + F_f_mag * t2 * t42 * t49 + F_f_mag * t4 * t49 * t78 + F_f_mag * t2 * t49 * t93
            + F_s_mag * t10 * t46 * t70 + F_s_mag * t9 * t50 * t73 + F_s_mag * t5 * t49 * t95
            - m_c * m_f * t27 * t81 * 2.0 + m_c * m_f * t27 * t130
            + F_f_mag * K_p * t2 * t71 + F_f_mag * K_p * t2 * t128
            + F_s_mag * K_p * t5 * t180 + F_f_mag * m_s * t4 * t50 * t51
            + F_f_mag * m_s * t2 * t50 * t56 + F_s_mag * m_s * t3 * t10 * t50
            + F_f_mag * t2 * t24 * t26 * t50 + F_f_mag * t4 * t24 * t27 * t49
            + F_s_mag * t3 * t20 * t26 * t50 + F_s_mag * K_p * m_s * t3 * t50 * w
            + F_f_mag * m_c * m_f * t4 * t27 * t49
    )

    # phase_X_u calculation using arctan2
    phase_X_u = np.arctan2(-t1517 * t1527, t1517 * (et6 + et7 + et8))

    # mag_X_f
    mag_X_f = np.sqrt(
        (
                t47 * t1489
                - F_f_mag * t1386 * t1407 * t1421
                - F_s_mag * t25 * t523 * t1395 * t1421
                + F_f_mag * t1383 * t1395 * t1421 * w
                + F_s_mag * t1117 * t1407 * t1421 * w
        ) ** 2
        + (-t1434 + t1436 - t1443 + t1453 + t1491) ** 2
    )

    # phase_X_f
    phase_X_f = np.arctan2(
        -t47 * t1489
        + F_f_mag * t1386 * t1407 * t1421
        + F_s_mag * t25 * t523 * t1395 * t1421
        - F_f_mag * t1383 * t1395 * t1421 * w
        - F_s_mag * t1117 * t1407 * t1421 * w,
        t1434 - t1436 + t1443 - t1453 - t1491
    )
    # mag_X_s
    mag_X_s = np.sqrt(
        (
                t47 * t1494
                + F_s_mag * t1385 * t1407 * t1421
                + F_f_mag * t25 * t522 * t1395 * t1421
                - F_f_mag * t1116 * t1407 * t1421 * w
                - F_s_mag * t1382 * t1395 * t1421 * w
        ) ** 2
        + (-t1433 + t1437 - t1442 + t1454 + t1495) ** 2
    )

    # phase_X_s
    phase_X_s = np.arctan2(
        t47 * t1494
        + F_s_mag * t1385 * t1407 * t1421
        + F_f_mag * t25 * t522 * t1395 * t1421
        - F_f_mag * t1116 * t1407 * t1421 * w
        - F_s_mag * t1382 * t1395 * t1421 * w,
        t1433 - t1437 + t1442 - t1454 - t1495
    )

    return mag_U, phase_U, real_P, imag_P, mag_X_u, phase_X_u, mag_X_f, phase_X_f, mag_X_s, phase_X_s