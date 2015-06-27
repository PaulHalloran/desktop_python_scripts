import numpy as np
import matplotlib.pyplot as plt
import random


initial_no_hawks = 100
initial_no_doves = 100

number_of_rounds = 3000

initial_energy = 100;

min_amount_of_food_available = 20
max_amount_of_food_available = 70

energy_required_for_reproduction = 250
#energy_cost_of_bluffing = 10
energy_loss_from_fighting = 200
energy_required_for_living = 1
energy_level_after_breeding = 125

record_of_hawk_numbers = [initial_no_hawks]
record_of_dove_numbers = [initial_no_doves]

hawks_energy_array = np.ones(initial_no_hawks)*initial_energy
doves_energy_array = np.ones(initial_no_doves)*initial_energy

for round in np.arange(number_of_rounds):

	amount_of_food_available = random.randint(min_amount_of_food_available,max_amount_of_food_available)

	#all birds expend some energy to live
	hawks_energy_array = hawks_energy_array - energy_required_for_living
	doves_energy_array = doves_energy_array - energy_required_for_living

	total_no_birds = np.size(hawks_energy_array)+np.size(doves_energy_array)

	if total_no_birds == 0:
		print 'sorry all your birds have died...'
		break


	selected_bird_1 = random.randint(0,total_no_birds-1)
	selected_bird_2 = random.randint(0,total_no_birds-1)
	bird_1 = 'hawk'
	bird_2 = 'hawk'

	number_of_hawks_at_end_of_round = np.size(hawks_energy_array)
	number_of_dove2_at_end_of_round = np.size(hawks_energy_array)

	if selected_bird_1 > np.size(hawks_energy_array) -1:
		bird_1 = 'dove'

	if selected_bird_2 > np.size(hawks_energy_array) -1:
		bird_2 = 'dove'

	if ((bird_1 == 'hawk') & (bird_2 == 'hawk')):
		#bird 1 wins the fight and gets food so gains energy
		hawks_energy_array[selected_bird_1] = hawks_energy_array[selected_bird_1] + amount_of_food_available
		#bird 2 looses the fight and is injured so looses energy
		hawks_energy_array[selected_bird_2] = hawks_energy_array[selected_bird_2] - energy_loss_from_fighting

	if ((bird_1 == 'dove') & (bird_2 == 'dove')):
		#both birds share the food
		doves_energy_array[selected_bird_1 - number_of_hawks_at_end_of_round] = doves_energy_array[selected_bird_1 - number_of_hawks_at_end_of_round] + amount_of_food_available/2.0
		#bird 2 looses the fight and is injured so looses energy
		doves_energy_array[selected_bird_2 - number_of_hawks_at_end_of_round] = doves_energy_array[selected_bird_2 - number_of_hawks_at_end_of_round] + amount_of_food_available/2.0

	if ((bird_1 == 'hawk') & (bird_2 == 'dove')):
		#the dove flies away leaving the hawk to eat all of the food
		hawks_energy_array[selected_bird_1] = hawks_energy_array[selected_bird_1] + amount_of_food_available

	if ((bird_1 == 'dove') & (bird_2 == 'hawk')):
		#the dove flies away leaving the hawk to eat all of the food
		hawks_energy_array[selected_bird_2] = hawks_energy_array[selected_bird_2] + amount_of_food_available

	#Do any of the hawks have enough anergy to reproduce?
	enough_energy_to_breed = np.where(hawks_energy_array > energy_required_for_reproduction)
	number_breeding = np.size(enough_energy_to_breed)
	if number_breeding > 0:
		#make some baby hawks
		hawks_energy_array = np.append(hawks_energy_array,np.ones(number_breeding)*initial_energy)
		#subtract the energy cost of breeding
		hawks_energy_array[enough_energy_to_breed] = energy_level_after_breeding
	
	#Do any of the hawks have enough anergy to reproduce?
	enough_energy_to_breed = np.where(doves_energy_array > energy_required_for_reproduction)
	number_breeding = np.size(enough_energy_to_breed)
	if number_breeding > 0:
		#make some baby doves
		doves_energy_array = np.append(doves_energy_array,np.ones(number_breeding)*initial_energy)
		#subtract the energy cost of breeding
		doves_energy_array[enough_energy_to_breed] = energy_level_after_breeding
	
	#Have and of the hawks run out of energy so die?
	dying_birds = np.where(hawks_energy_array < 1)
	number_dying = np.size(dying_birds)
	if number_dying > 0:
		#cull low-energy birds
		hawks_energy_array = np.delete(hawks_energy_array,dying_birds)
	
	#Have and of the doves run out of energy so die?
	dying_birds = np.where(doves_energy_array < 1)
	number_dying = np.size(dying_birds)
	if number_dying > 0:
		#cull low-energy birds
		doves_energy_array = np.delete(doves_energy_array,dying_birds)

	number_of_hawks_at_end_of_round = np.size(hawks_energy_array)
	number_of_doves_at_end_of_round = np.size(doves_energy_array)

	record_of_hawk_numbers = np.append(record_of_hawk_numbers,number_of_hawks_at_end_of_round)
	record_of_dove_numbers = np.append(record_of_dove_numbers,number_of_doves_at_end_of_round)

	
plt.close('all')
plt.plot(record_of_hawk_numbers,'r',label = 'number of hawks', linewidth = 3)	
plt.plot(record_of_dove_numbers,'b',label = 'number of doves', linewidth = 3)
plt.xlabel('time')
plt.ylabel('no. birds')
plt.legend()
plt.show()
