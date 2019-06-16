from keras.layers import Lambda, Concatenate
# from sklearn.svm import

def slice(x, start=None, end=None):
    return x[:, start:end]


def cfenli(c, i):
    ci = Lambda(slice, arguments={'start': i, 'end': i + 1})(c)
    c_front_i = Lambda(slice, arguments={'end': i})(c)
    c_back_i = Lambda(slice, arguments={'start': i + 1})(c)
    cnoi = Concatenate()([c_front_i, c_back_i])
    return ci, cnoi

print(int(1200%1e3))
