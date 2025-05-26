from face_recognizer import FaceRecognizer
from knowledge_graph.graph_manager import GraphManager
from dialog.fortune_teller import FortuneTeller
from tiago_interface.tiago_controller import TiagoController

def main():
    face_rec = FaceRecognizer()
    graph = GraphManager()
    fortune = FortuneTeller()
    tiago = TiagoController()

    tiago.say("Welcome to the Tiago Fortune Teller!")
    image = tiago.capture_image()
    person_id = "user1"
    is_new = face_rec.add_face(image, person_id)
    if is_new:
        tiago.say("Hello new friend!")
    graph.update(person_id, "visited_fortune_teller")
    embedding = graph.get_embedding(person_id)
    response = fortune.generate(person_id, embedding)
    tiago.gesture("mystical_wave")
    tiago.say(response)

if __name__ == "__main__":
    main()
