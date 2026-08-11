LG_MODES = [
    (0,0), (0,1), (0,2), 
    (1,0), (1,1), (1,2), 
    (2,0), (2,1), (2,2)
]

WAVELENGTH = 529e-9

SENSOR_SIZE = (4512, 4576)

PIXEL_SIZE = 2.74e-6

CENTER_SHIFT = 50

EXPOSURE = 5e-6

QE = 0.68

READ_NOISE = 2.3

DARK_CURRENT = 22

FULL_WELL = 9_400

ADC_BITS = 12

POWER_RANGE = (FULL_WELL//2, FULL_WELL)

W0_RANGE = (200e-6, 800e-6)

SEED = None

L = 2e-3
# Create bigger image -> shifting realized by cropping the image
N = max(SENSOR_SIZE) + 2 * CENTER_SHIFT

N_IMAGES_PER_CLASS = 1
