import random

class HSMA:
    def __init__(self, name, role, py_confidence):
        self.name = name
        self.role = role
        self.py_confidence = py_confidence

        self.psg = random.choice(["Atari", "Commodore", "Sinclair", "Amstrad",
                                  "Acorn"])
        
        self.self_confidence = random.uniform(0.5, 0.9)
        
    def learn_python(self, num_hours):
        for hour in range(num_hours):
            if random.uniform(0, 1) < self.self_confidence:
                if self.py_confidence < 100:
                    self.py_confidence += 1
            else:
                if self.py_confidence > 0:
                    self.py_confidence -= 1

class Trainer:
    def __init__(self, name, teaching_quality):
        self.name = name
        self.teaching_quality = teaching_quality

    def run_support_session(self, participants):
        for participant in participants:
            if random.uniform(0, 1) < self.teaching_quality:
                if participant.self_confidence < 0.9:
                    participant.self_confidence += 0.01

                if participant.py_confidence < 100:
                    participant.py_confidence += 1

list_of_hsmas = []
list_of_trainers = []

hsma_jessica = HSMA("Jessica Fletcher", "Author", 60)
list_of_hsmas.append(hsma_jessica)

hsma_seth = HSMA("Seth Hazlitt", "Doctor", 30)
list_of_hsmas.append(hsma_seth)

hsma_amos = HSMA("Amos Tupper", "Sheriff", 10)
list_of_hsmas.append(hsma_amos)

trainer_dan = Trainer("Dan Chalk", 0.4)
list_of_trainers.append(trainer_dan)

trainer_sammi = Trainer("Sammi Rosser", 0.8)
list_of_trainers.append(trainer_sammi)

for hsma in list_of_hsmas:
    print (f"Name : {hsma.name}")
    print (f"Role : {hsma.role}")
    print (f"PSG : {hsma.psg}")
    print (f"Python Confidence : {hsma.py_confidence}")
    print (f"Self Confidence : {hsma.self_confidence:.2f}")
    print ()

for hsma in list_of_hsmas:
    hsma.learn_python(10)

print ("Update after HSMAs are learning Python")
print ("---")

for hsma in list_of_hsmas:
    print (f"Name : {hsma.name}")
    print (f"Role : {hsma.role}")
    print (f"PSG : {hsma.psg}")
    print (f"Python Confidence : {hsma.py_confidence}")
    print (f"Self Confidence : {hsma.self_confidence:.2f}")
    print ()

for trainer in list_of_trainers:
    print (f"{trainer.name} is running support sessions")
    for session in range(20):
        trainer.run_support_session(list_of_hsmas)

    print ("Update :")
    print ("---")

    for hsma in list_of_hsmas:
        print (f"Name : {hsma.name}")
        print (f"Role : {hsma.role}")
        print (f"PSG : {hsma.psg}")
        print (f"Python Confidence : {hsma.py_confidence}")
        print (f"Self Confidence : {hsma.self_confidence:.2f}")
        print ()

