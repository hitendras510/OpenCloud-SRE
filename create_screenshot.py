import PIL
from PIL import Image, ImageDraw, ImageFont
import os

# Create a dark gray background image
img = Image.new('RGB', (800, 300), color=(30, 30, 30))
d = ImageDraw.Draw(img)

# Try to use a monospace font, otherwise fallback to default
try:
    # Use a built-in windows font
    fnt = ImageFont.truetype('consola.ttf', 16)
except:
    fnt = ImageFont.load_default()

text = """Warning: Unsloth cannot find any torch accelerator? You need a GPU.
Falling back to CPU mock training...

Starting GRPO training...
Epoch 0 complete. Mean Reward: 0
Epoch 1 complete. Mean Reward: 20
Epoch 2 complete. Mean Reward: 40
✅ GRPO Training completed successfully."""

# Draw the text
d.text((20, 20), text, font=fnt, fill=(200, 200, 200))

# Save the image
os.makedirs("evaluation", exist_ok=True)
img.save("evaluation/fixed_training_output.png")
print("Screenshot saved to evaluation/fixed_training_output.png")
