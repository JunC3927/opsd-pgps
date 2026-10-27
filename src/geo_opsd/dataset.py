"""Keep raw RGB images identical to the length probe and the teacher input.

Qwen3-VL's own image processor performs resizing, normalization and patchification.
The generic verl dataset pre-resizes with qwen_vl_utils; avoid that second resize.
"""
from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset

class PGPSDataset(RLHFDataset):
    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        images = []
        for message in messages:
            for item in message['content']:
                if item['type'] == 'image':
                    source = item.get('image', item.get('path'))
                    if isinstance(source, Image.Image):
                        images.append(source.convert('RGB'))
                    else:
                        with Image.open(source) as im:
                            images.append(im.convert('RGB'))
                elif item['type'] == 'video':
                    raise ValueError('PGPS9K does not contain video inputs')
        assert len(images) == 1
        return images, None
