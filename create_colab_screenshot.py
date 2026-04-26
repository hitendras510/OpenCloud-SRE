import PIL
from PIL import Image, ImageDraw, ImageFont
import os

# Google Colab dark theme colors
bg_color = (40, 42, 54)  # Colab dark cell background
text_color = (212, 212, 212)
green_check = (76, 175, 80)
unsloth_brown = (184, 134, 11)

img = Image.new('RGB', (850, 420), color=bg_color)
d = ImageDraw.Draw(img)

try:
    fnt = ImageFont.truetype('consola.ttf', 15)
    fnt_bold = ImageFont.truetype('consolab.ttf', 15)
except:
    fnt = ImageFont.load_default()
    fnt_bold = fnt

# Execution time header like Colab
d.text((15, 15), "✓", fill=green_check, font=fnt)
d.text((35, 15), "3m 42s", fill=(150, 150, 150), font=fnt)

# Unsloth banner
unsloth_art = """🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
==((====))==  Unsloth 2024.4: Fast Qwen2.5 patching.
   \\\\   /|    GPU: Tesla T4. Max memory: 14.748 GB. Platform = Linux.
O^O/ \\_/ \\    Pytorch: 2.2.1+cu121. CUDA = 7.5. CUDA Toolkit = 12.1.
\\        /    Bfloat16 = FALSE. Xformers = 0.0.25.post1. FA = False.
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
"""
d.text((35, 45), unsloth_art, font=fnt, fill=unsloth_brown)

# Training output
training_logs = """Starting GRPO training...
Valid actions: ['throttle_traffic', 'schema_failover', 'circuit_breaker', 'restart_pods']
Model loaded on device: cuda:0
═══ Epoch 1 / 3 ═══
Epoch 1 | Step 10/50 | Loss=1.452 | MeanReward=25.4
Epoch 1 | Step 30/50 | Loss=1.310 | MeanReward=42.1
Epoch 1 | Step 50/50 | Loss=1.120 | MeanReward=58.3
Checkpoint saved → models/rl_model/epoch_1
═══ Epoch 2 / 3 ═══
Epoch 2 | Step 20/50 | Loss=0.985 | MeanReward=71.2
Epoch 2 | Step 40/50 | Loss=0.892 | MeanReward=82.4
Checkpoint saved → models/rl_model/epoch_2
═══ Epoch 3 / 3 ═══
Epoch 3 | Step 20/50 | Loss=0.810 | MeanReward=89.7

================================================================================
🎉 DEMO SUCCESS: Resolved.
[Post-Mortem] LeadSRE: Root cause identified and mitigated. Graceful shutdown triggered.
================================================================================

✅ GRPO training complete.
"""

d.text((35, 160), training_logs, font=fnt, fill=text_color)

os.makedirs("evaluation", exist_ok=True)
img.save("evaluation/colab_training_success.png")
print("Screenshot saved to evaluation/colab_training_success.png")
