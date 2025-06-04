#!/usr/bin/env python3

import cv2
import sys
import os
import rospy

# Aggiungi il percorso src al Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_recognition.face_recognizer import OpenCVFaceRecognizer

def capture_and_add_face(recognizer, person_id, person_name=None):
    """
    Cattura una foto dalla webcam e aggiunge il volto al database.
    
    Args:
        recognizer: Instance del riconoscitore facciale
        person_id: ID unico della persona
        person_name: Nome leggibile della persona (opzionale)
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam")
        return False
    
    print(f"\nAggiungo il volto per: {person_id}")
    if person_name:
        print(f"Nome: {person_name}")
    print("Premi SPAZIO per catturare, ESC per annullare")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore nella cattura del frame")
            break
        
        # Rileva volti nel frame corrente
        faces = recognizer.detect_faces(frame)
        
        # Disegna rettangoli sui volti rilevati
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Face Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spazio per catturare
            if len(faces) > 0:
                success = recognizer.add_face(frame, person_id, person_name)
                if success:
                    print(f"Volto aggiunto con successo per {person_id}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Errore nell'aggiunta del volto")
            else:
                print("Nessun volto rilevato, riprova")
        elif key == 27:  # ESC per uscire
            print("Operazione annullata")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def add_face_from_file(recognizer, image_path, person_id, person_name=None):
    """
    Aggiunge un volto da un file immagine.
    
    Args:
        recognizer: Instance del riconoscitore facciale
        image_path: Percorso del file immagine
        person_id: ID unico della persona
        person_name: Nome leggibile della persona (opzionale)
    """
    if not os.path.exists(image_path):
        print(f"Errore: file {image_path} non trovato")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Errore: impossibile leggere l'immagine {image_path}")
        return False
    
    success = recognizer.add_face(image, person_id, person_name)
    if success:
        print(f"Volto aggiunto con successo per {person_id} da {image_path}")
        return True
    else:
        print(f"Errore nell'aggiunta del volto da {image_path}")
        return False

def test_recognition(recognizer):
    """
    Testa il riconoscimento facciale con la webcam.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam")
        return
    
    print("\nModalità test riconoscimento - Premi ESC per uscire")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Riconosci volti
        results = recognizer.recognize_face(frame)
        
        # Visualizza risultati
        if results:
            annotated_frame = recognizer.visualize_recognition_results(frame, results)
            cv2.imshow('Face Recognition Test', annotated_frame)
        else:
            cv2.imshow('Face Recognition Test', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Inizializza ROS (facoltativo per questo script)
    rospy.init_node('face_database_populator', log_level=rospy.INFO)
    
    # Inizializza il riconoscitore
    recognizer = OpenCVFaceRecognizer()
    
    while True:
        print("\n" + "="*50)
        print("POPOLAMENTO DATABASE RICONOSCIMENTO FACCIALE")
        print("="*50)
        
        # Mostra statistiche attuali
        stats = recognizer.get_database_stats()
        print(f"Persone nel database: {stats['total_people']}")
        print(f"Campioni di training: {stats['training_samples']}")
        print(f"Riconoscimenti totali: {stats['total_recognitions']}")
        print(f"Modello addestrato: {'Sì' if stats['is_trained'] else 'No'}")
        
        if stats['total_people'] > 0:
            print("\nPersone conosciute:")
            for person_id in recognizer.get_all_known_people():
                info = recognizer.get_person_info(person_id)
                name = info.get('name', 'N/A')
                count = info.get('recognition_count', 0)
                print(f"  - {person_id} ({name}) - Riconoscimenti: {count}")
        
        print("\nOpzioni:")
        print("1. Aggiungi volto dalla webcam")
        print("2. Aggiungi volto da file immagine")
        print("3. Testa riconoscimento con webcam")
        print("4. Mostra statistiche")
        print("5. Esci")
        
        choice = input("\nScegli un'opzione (1-5): ").strip()
        
        if choice == '1':
            person_id = input("Inserisci l'ID della persona: ").strip()
            person_name = input("Inserisci il nome (opzionale): ").strip() or None
            
            if person_id:
                capture_and_add_face(recognizer, person_id, person_name)
            else:
                print("ID persona richiesto!")
        
        elif choice == '2':
            image_path = input("Inserisci il percorso dell'immagine: ").strip()
            person_id = input("Inserisci l'ID della persona: ").strip()
            person_name = input("Inserisci il nome (opzionale): ").strip() or None
            
            if image_path and person_id:
                add_face_from_file(recognizer, image_path, person_id, person_name)
            else:
                print("Percorso immagine e ID persona richiesti!")
        
        elif choice == '3':
            if stats['is_trained']:
                test_recognition(recognizer)
            else:
                print("Nessun modello addestrato disponibile. Aggiungi prima alcuni volti.")
        
        elif choice == '4':
            # Le statistiche sono già mostrate sopra
            pass
        
        elif choice == '5':
            print("Arrivederci!")
            break
        
        else:
            print("Opzione non valida!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente")
    except Exception as e:
        print(f"Errore: {e}")