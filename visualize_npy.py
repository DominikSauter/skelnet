from PIL import Image
import numpy as np
import click


@click.command()
@click.option('--in_pts', default='npy/in_pts.npy')
@click.option('--out_pts', default='npy/out_pts.npy')
@click.option('--example', default=0)
def main(in_pts, out_pts, example):
    in_pix = np.load(in_pts)
    out_pix = np.load(out_pts)

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
    Image.fromarray(in_pix[example, :, :, 0], 'L').show()
    Image.fromarray(out_pix[example, :, :, 0], 'L').show()


if __name__ == '__main__':
    main()

