from PIL import Image
import numpy as np


def main():
    in_pix = np.load('npy/in_train_ml_comp.npy')
    out_pix = np.load('npy/out_train_ml_comp.npy')

    #in_img = Image.new('1', (255, 255))
    #out_img = Image.new('1', (255, 255))
    #in_immat = in_img.load()
    #out_immat = out_img.load()
    #for i in range(255):
    #    for j in range(255):
    #        in_immat[(i, j)] = int(in_pix[0, i, j, 0])
    #        out_immat[(i, j)] = int(out_pix[0, i, j, 0])
    #in_img.save('npy/in_pix.png')
    #out_img.save('npy/out_pix.png')

    print(in_pix.shape)
    print(out_pix.shape)
    example = 2
    Image.fromarray(in_pix[example, :, :, 0], 'L').show()
    Image.fromarray(out_pix[example, :, :, 0], 'L').show()


if __name__ == '__main__':
    main()

