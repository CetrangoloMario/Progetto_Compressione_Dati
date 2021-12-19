import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math as math
import scipy.fft
import confresnel1D
import fourierfresnel1D
from givensphaserot import givenphaserot



def intfresnel2D(x, fw, pp, z, wlen, algo):
    # aggiungere un assert per controllare se l'array x non contiene zero
    assert (np.size(x) % 2) == 0, "Hologram dimensions must be even"
    assert np.size(x, 0) == np.size(x, 1), "Hologram must be square (to be resolved in next version)"

    def fw_fresnel(r):
        return np.round(fpropfun(r, pp, fz, wlen))

    def bw_fresnel(r):
        return np.round(bpropfun(r, pp, bz, wlen))

    def fwdfourierfresnel1D(x, pp, z, wlen):
        return fourierfresnel1D.fourierfresnel1D(x, pp, z, wlen, True)

    def revfourierfresnel1D(x, pp, z, wlen):
        return fourierfresnel1D.fourierfresnel1D(x, pp, z, wlen, False)

    def apply_transform():
        if (fw):
            o = np.conjugate(x[:, 1::2])
            e = x[:, 0::2]

            e = e + bw_fresnel(o)
            o = o - fw_fresnel(e)
            e = e + bw_fresnel(o)

            x[:, 0::2] = -o
            x[:, 1::2] = np.conjugate(e)

            return x
        else:
            e = np.conj(x[:, 1::2])  # righe le prende tutte, colonne prende a 2 a 2
            o = -x[:, 0::2]  # righe le prende tutte, colonne parte dalla prima  e a step di 2

            e = e - bw_fresnel(o)
            o = o + fw_fresnel(e)
            e = e - bw_fresnel(o)

            x[:, 0::2] = e
            x[:, 1::2] = np.conj(o)
            return x

    if algo == "conv":
        fpropfun = confresnel1D.convfresnel1D  # convfresnel1D(x,pp,z,wlen)
        bpropfun = confresnel1D.convfresnel1D  # convfresnel1D(x,pp,z,wlen)
        fz = z
        bz = -z
    elif algo == "four":
        fpropfun = fwdfourierfresnel1D
        fz = z
        bpropfun = revfourierfresnel1D
        bz = z
    else:
        print("algoritmo sconosciuto")

    for i in [-1, 1]:
        if fw:
            x = np.rot90(x, i)
        x = apply_transform()
        if not fw:
            x = np.rot90(x, i)
    return x



def fourier_phase_multiply(r, fw, pp, z, wlen):
    n = np.size(r, 0)
    xx = np.power(np.matrix(np.arange(-n / 2, n / 2)), 2)
    ppout = wlen * np.abs(z) / n / pp

    temp = (math.pi * np.power(ppout, 2)) /((wlen * np.abs(z)))
    p=np.multiply(temp, (xx + xx.transpose()))+ (2 * np.pi * z )/ wlen
    return givenphaserot(r,p, fw)



def main():
    f = scipy.io.loadmat('exampleholo.mat')  # aprire il file .mat
    pp = 8e-6  # pixel pitch
    pp = np.matrix(pp)
    wlen = 532e-9  # wavelenght
    wlen = np.matrix(wlen)
    dist = 5e-1  # propogation depth
    dist = np.matrix(dist)

    holo = np.matrix(f.get("holo"))  # estraggo dal file .mat solo holo e lo converto da np a array


    # plt.imshow(np.real(holo), cmap="gray")

    # plt.show()  # mostro l'ologramma pre - compressione
    # algo = "conv"
    algo = 'four';
    t = intfresnel2D(np.csingle(holo), False, pp, dist, wlen, algo)
    # res = np.real(t)
    # print("res: ",res[255,254])
    plt.show()
    t = fourier_phase_multiply(t, False, pp, dist, wlen)
    plt.imshow(np.real(t), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
    # prova()
