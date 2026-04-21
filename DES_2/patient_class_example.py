import random

class Patient:
    def __init__(self, p_id):
        self.p_id = p_id

        self.priority = random.randint(1,5)
        self.previous_admission_60_days = random.choice([True, False])

    def progress_condition(self, probability):
        if random.uniform(0,1) < probability:
            if self.priority > 1:
                self.priority -= 1
    
    def choose_ed_to_attend(self, options):
        chosen_ed = random.choice(options)

        return chosen_ed
    
patient_dan = Patient(165)
patient_sammi = Patient(321)

dans_chosen_ed = patient_dan.choose_ed_to_attend(["Plymouth", "Truro"])
print (f"Dan chose to go to {dans_chosen_ed}")

sammis_chosen_ed = patient_sammi.choose_ed_to_attend(["Exeter", "Torbay"])
print (f"Sammi chose to go to {sammis_chosen_ed}")

print (f"Dan began at priority {patient_dan.priority}")
patient_dan.progress_condition(0.8)
print (f"After an hour, Dan was at priority {patient_dan.priority}")

print (f"Sammi began at priority {patient_sammi.priority}")
patient_sammi.progress_condition(0.1)
print (f"After an hour, Sammi was at priority {patient_sammi.priority}")

