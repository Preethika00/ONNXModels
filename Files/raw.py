import numpy as np

dummy_inputs={
    "pixel_values": np.random.randn(1, 3, 768, 768).astype(np.float32),
    "input_ids": np.random.randn(80,16).astype(np.int64),
    "attention_mask": np.random.randn(80, 16).astype(np.int64),
}
dummy_inputs["pixel_values"].tofile("/media/ava/DATA2/preethika/Aimet/DSP/int8/input_image.raw")
dummy_inputs["input_ids"].tofile("/media/ava/DATA2/preethika/Aimet/DSP/int8/input_ids.raw")
dummy_inputs["attention_mask"].tofile("/media/ava/DATA2/preethika/Aimet/DSP/int8/mask.raw")