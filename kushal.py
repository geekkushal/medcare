import ml as kushal

print(" altered_sensorium , weakness_of_one_body_side , headache -> Brain haemorage ")
print()
print(" abdominal_pain , dark_urine , yellowish_skin , high_fever , weight_loss , fatigue , vomiting , itching -> jaundice")
print()
print("red_spots_over_body , muscle_pain , loss_of_appetite , pain_behind_the_eyes , back_pain , headache , high_fever , fatigue , chills , joint_pain , skin_rash")
print()
print()

n = int(input(" Enter number of symptoms: "))
sym = []
for i in range (n):
    t = input()
    sym.append(t)
print(kushal.predict_disease_by_kushal(sym))
