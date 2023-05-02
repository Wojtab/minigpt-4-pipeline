from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class Blip2ImageEvalProcessor():
    def __init__(self, image_size=224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        self.normalize = transforms.Normalize(mean, std)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)
