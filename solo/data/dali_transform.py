import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class RandomGrayScaleConversion:
    def __init__(self, prob: float = 0.2, device: str = "gpu"):
        self.prob = prob
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            out = self.grayscale(images)
            out = fn.cat(out, out, out, axis=2)
        else:
            out = images
        return out


class RandomRotate:
    def __init__(self, prob: float = 0.5, angle1=-45, angle2=45, device: str = "gpu"):
        self.prob = prob
        self.angle1 = angle1
        self.angle2 = angle2

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            angle = fn.random.uniform(range=(self.angle1, self.angle2))
            out = fn.rotate(images, angle=angle, fill_value=0, keep_size=True)
        else:
            out = images
        return out


class RandomEqualize:
    def __init__(self, prob: float = 0.5, device: str = "gpu"):
        self.prob = prob
        self.equalize = ops.experimental.Equalize(device=device,)

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            out = self.equalize(images)
        else:
            out = images
        return out


class RandomColorJitter:
    def __init__(
        self, brightness: float, contrast: float, saturation: float, hue: float,  prob: float = 0.8, device: str = "gpu",
    ):
        assert 0 <= hue <= 0.5

        self.prob = prob
        self.color = ops.ColorTwist(device=device)

        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        if brightness: self.brightness = ops.random.Uniform(range=[max(0, 1 - brightness), 1 + brightness])
        if contrast: self.contrast = ops.random.Uniform(range=[max(0, 1 - contrast), 1 + contrast])
        if saturation: self.saturation = ops.random.Uniform(range=[max(0, 1 - saturation), 1 + saturation])

        if hue:
            hue = 360 * hue
            self.hue = ops.random.Uniform(range=[-hue, hue])

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            out = self.color(
                images,
                brightness=self.brightness() if callable(self.brightness) else self.brightness,
                contrast=self.contrast() if callable(self.contrast) else self.contrast,
                saturation=self.saturation() if callable(self.saturation) else self.saturation,
                hue=self.hue() if callable(self.hue) else self.hue,
            )
        else:
            out = images
        return out


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.5, window_size: int = 23, device: str = "gpu"):
        self.prob = prob
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(window_size, window_size))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            sigma = self.sigma() * 1.9 + 0.1
            out = self.gaussian_blur(images, sigma=sigma)
        else:
            out = images
        return out


class RandomSolarize:
    def __init__(self, threshold: int = 128, prob: float = 0.0):

        self.prob = prob
        self.threshold = threshold

    def __call__(self, images):
        do_op = fn.random.coin_flip(probability=self.prob, dtype=types.DALIDataType.BOOL)
        if do_op:
            inverted_img = types.Constant(255, dtype=types.UINT8) - images
            mask = images >= self.threshold
            out = mask * inverted_img + (True ^ mask) * images
        else:
            out = images
        return out