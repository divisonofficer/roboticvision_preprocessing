import torch
import cv2
import numpy as np


def fusion(V: cv2.typing.MatLike, NIR: cv2.typing.MatLike, mask: cv2.typing.MatLike):
    Gv = cv2.cvtColor(V, cv2.COLOR_BGR2GRAY)
    if not mask is None:
        Gv = cv2.bitwise_and(Gv, mask)
        NIR = cv2.bitwise_and(NIR, mask)
    Gv = Gv.astype(np.float64)
    NIR = NIR.astype(np.float64)
    # Gv = (Gv - Gv.mean()) / (np.sqrt(NIR.var() / Gv.var())) + NIR.mean()
    # Gv = cv2.normalize(Gv, None, 0, 255, cv2.NORM_MINMAX)
    NIR = cv2.normalize(NIR, None, 0, Gv.max(), cv2.NORM_MINMAX)
    Gvn = cv2.normalize(Gv - NIR, None, 0, 1, cv2.NORM_MINMAX)

    print(Gv.min(), Gv.max(), NIR.min(), NIR.max(), Gvn.min(), Gvn.max())

    YCB = cv2.cvtColor(V, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    I = YCB[:, :, 0]
    alpha = YCB[:, :, 1]
    beta = YCB[:, :, 2]
    #
    NIR = cv2.normalize(NIR, None, 0, I.max(), cv2.NORM_MINMAX)

    I_t = I * Gvn + NIR * (1 - Gvn)
    M = (I - I_t) / I
    M = M.clip(-10, 1)

    # I_t = (I_t - I_t.mean()) * (np.sqrt(I.var() / I_t.var())) + I.mean()
    print("M", M.min(), M.max(), M.mean())
    print("I", I.min(), I.max(), I.mean())
    print("I_t", I_t.min(), I_t.max(), I_t.mean())
    I_t_I = I_t / I
    print("I_t_I", I_t_I.min(), I_t_I.max(), I_t_I.mean())
    M.clip(-3, 1, out=M)
    Alpha_T = M * alpha + alpha
    Beta_T = M * beta + beta
    print("Alpha_Pre", alpha.min(), alpha.max(), alpha.mean(), alpha.var())
    print(Alpha_T.max(), Alpha_T.min(), Alpha_T.mean(), Alpha_T.var())
    print("Beta_Pre", beta.min(), beta.max(), beta.mean(), beta.var())
    print(Beta_T.max(), Beta_T.min(), Beta_T.mean(), Beta_T.var())
    Alpha_T = Alpha_T * (alpha.mean() / Alpha_T.mean())
    Beta_T = Beta_T * (beta.mean() / Beta_T.mean())
    print("Alpha", Alpha_T.min(), Alpha_T.max(), Alpha_T.mean(), Alpha_T.var())
    print("Beta", Beta_T.min(), Beta_T.max(), Beta_T.mean(), Beta_T.var())

    print("")

    IMG = cv2.merge((I_t, Alpha_T, Beta_T)).astype(np.uint8)
    return (
        cv2.bitwise_and(
            cv2.cvtColor(IMG, cv2.COLOR_YCrCb2BGR),
            np.repeat(mask[:, :, np.newaxis], 3, axis=2),
        )
        if not mask is None
        else cv2.cvtColor(IMG, cv2.COLOR_YCrCb2BGR)
    )
