class TiagoController:
    def say(self, text):
        print(f"Tiago says: {text}")

    def gesture(self, type_):
        print(f"Tiago performs gesture: {type_}")

    def capture_image(self):
        # Dummy placeholder: return a blank numpy array
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)
