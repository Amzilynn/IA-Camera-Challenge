try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    fer_model = HSEmotionRecognizer(model_name='resnet34_7_224_combined', device='cpu')
    print("SUCCESS: HSEmotion initialized.")
except Exception as e:
    print(f"FAILURE: {e}")
