import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

'''
by Raefah Wahid
'''

data = pd.read_csv('emods_data.csv')

# print(data)

print("efficacy and performance")
# efficacy and performance
quiz1_total = data['Q1_Total']
s1_total = data['S1_Total']
efficacy_on_performance_before = stats.pearsonr(s1_total, quiz1_total)
print(efficacy_on_performance_before)
slope, intercept, r_value, p_value, std_err = stats.linregress(s1_total, quiz1_total)

y = slope.item()*s1_total + intercept.item()
plt.title('Self-Efficacy vs. Performance')
plt.xlabel('self-efficacy (survey 1)', fontsize=18)
plt.ylabel('performance (quiz 1)', fontsize=18)
plt.scatter(s1_total, quiz1_total, c='cyan')
plt.plot(s1_total, y, 'r')
plt.show()

quiz2_total = data['Q2_Total']
s2_total = data['S2_Total']
efficacy_on_performance_after = stats.pearsonr(s2_total, quiz2_total)
print(efficacy_on_performance_after)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(s2_total, quiz2_total)


z = slope2.item()*s2_total + intercept2.item()
plt.title('Self-Efficacy vs. Performance')
plt.xlabel('self-efficacy (survey 2)', fontsize=18)
plt.ylabel('performance (quiz 2)', fontsize=18)
plt.scatter(s2_total, quiz2_total, c='magenta')
plt.plot(s2_total, z, 'r', color='blue')
plt.show()


# experience
print("experience stats")
low_quiz1_total = []
medium_quiz1_total = []
high_quiz1_total = []
row = 0
for elt in data['Experience?']:
    if elt == 0:
        low_quiz1_total.append(data['Q1_Total'][row])
    elif elt == 1:
        medium_quiz1_total.append(data['Q1_Total'][row])
    else:
        high_quiz1_total.append(data['Q1_Total'][row])
    row += 1

# one-way ANOVA with experience against quiz1
experience_quiz1 = stats.f_oneway(low_quiz1_total, medium_quiz1_total, high_quiz1_total)
print(experience_quiz1)

plt.title('Experience vs. Performance')
plt.xlabel('experience levels', fontsize=18)
plt.ylabel('quiz 1 scores', fontsize=18)
plt.scatter(['Low']*(len(low_quiz1_total)), low_quiz1_total, c='cyan')
plt.scatter(['Medium']*(len(medium_quiz1_total)), medium_quiz1_total, c='magenta')
plt.scatter(['High']*(len(high_quiz1_total)), high_quiz1_total, c='green')
plt.show()

low_quiz2_total = []
medium_quiz2_total = []
high_quiz2_total = []
row = 0
for elt in data['Experience?']:
    if elt == 0:
        low_quiz2_total.append(data['Q2_Total'][row])
    elif elt == 1:
        medium_quiz2_total.append(data['Q2_Total'][row])
    else:
        high_quiz2_total.append(data['Q2_Total'][row])
    row += 1

# one-way ANOVA with experience against quiz2
experience_quiz2 = stats.f_oneway(low_quiz2_total, medium_quiz2_total, high_quiz2_total)
print(experience_quiz2)

plt.title('Experience vs. Performance')
plt.xlabel('experience levels', fontsize=18)
plt.ylabel('quiz 2 scores', fontsize=18)
plt.scatter(['Low']*(len(low_quiz2_total)), low_quiz2_total, c='cyan')
plt.scatter(['Medium']*(len(medium_quiz2_total)), medium_quiz2_total, c='magenta')
plt.scatter(['High']*(len(high_quiz2_total)), high_quiz2_total, c='green')
plt.show()


low_s1_total = []
medium_s1_total = []
high_s1_total = []
row = 0
for elt in data['Experience?']:
    if elt == 0:
        low_s1_total.append(data['S1_Total'][row])
    elif elt == 1:
        medium_s1_total.append(data['S1_Total'][row])
    else:
        high_s1_total.append(data['S1_Total'][row])
    row += 1

# one-way ANOVA with experience against s1
experience_s1 = stats.f_oneway(low_s1_total, medium_s1_total, high_s1_total)
print(experience_s1)

plt.title('Experience vs. Self-Efficacy')
plt.xlabel('experience levels', fontsize=18)
plt.ylabel('survey 1 scores', fontsize=18)
plt.scatter(['Low']*(len(low_s1_total)), low_s1_total, c='cyan')
plt.scatter(['Medium']*(len(medium_s1_total)), medium_s1_total, c='magenta')
plt.scatter(['High']*(len(high_s1_total)), high_s1_total, c='green')
plt.show()

low_s2_total = []
medium_s2_total = []
high_s2_total = []
row = 0
for elt in data['Experience?']:
    if elt == 0:
        low_s2_total.append(data['S2_Total'][row])
    elif elt == 1:
        medium_s2_total.append(data['S2_Total'][row])
    else:
        high_s2_total.append(data['S2_Total'][row])
    row += 1

# one-way ANOVA with experience against s2
experience_s2 = stats.f_oneway(low_s2_total, medium_s2_total, high_s2_total)
print(experience_s2)

plt.title('Experience vs. Self-Efficacy')
plt.xlabel('experience levels', fontsize=18)
plt.ylabel('survey 2 scores', fontsize=18)
plt.scatter(['Low']*(len(low_s2_total)), low_s2_total, c='cyan')
plt.scatter(['Medum']*(len(medium_s2_total)), medium_s2_total, c='magenta')
plt.scatter(['High']*(len(high_s2_total)), high_s2_total, c='green')
plt.show()

# gender

print("gender stats")
male_quiz1_total = []
female_quiz1_total = []
row = 0
for elt in data['Gender']:
    if elt == "Male":
        male_quiz1_total.append(data['Q1_Total'][row])
    else:
        female_quiz1_total.append(data['Q1_Total'][row])
    row += 1

# one-way ANOVA with gender against quiz1
gender_quiz1 = stats.f_oneway(male_quiz1_total, female_quiz1_total)
print(gender_quiz1)

one = np.arange(18)
plt.title('Gender vs. Performance')
plt.xlabel('gender', fontsize=18)
plt.ylabel('quiz 1 scores', fontsize=18)
plt.scatter(['Male']*(len(male_quiz1_total)), male_quiz1_total, c=one, cmap='plasma_r', marker = ">")
one = np.arange(29)
plt.scatter(['Female']*(len(female_quiz1_total)), female_quiz1_total, c=one, cmap='plasma_r', marker = ">")
plt.show()

male_quiz2_total = []
female_quiz2_total = []
row = 0
for elt in data['Gender']:
    if elt == "Male":
        male_quiz2_total.append(data['Q2_Total'][row])
    else:
        female_quiz2_total.append(data['Q2_Total'][row])
    row += 1

# one-way ANOVA with gender against quiz2
gender_quiz2 = stats.f_oneway(male_quiz2_total, female_quiz2_total)
print(gender_quiz2)

one = np.arange(18)
plt.title('Gender vs. Performance')
plt.xlabel('gender', fontsize=18)
plt.ylabel('quiz 2 scores', fontsize=18)
plt.scatter(['Male']*(len(male_quiz2_total)), male_quiz2_total, c=one, cmap='plasma_r', marker = ">")
one = np.arange(29)
plt.scatter(['Female']*(len(female_quiz2_total)), female_quiz2_total, c=one, cmap='plasma_r', marker = ">")
plt.show()

male_s1_total = []
female_s1_total = []
row = 0
for elt in data['Gender']:
    if elt == "Male":
        male_s1_total.append(data['S1_Total'][row])
    else:
        female_s1_total.append(data['S1_Total'][row])
    row += 1

# one-way ANOVA with gender against s1
gender_s1 = stats.f_oneway(male_s1_total, female_s1_total)
print(gender_s1)

one = np.arange(18)
plt.title('Gender vs. Self-Efficacy')
plt.xlabel('gender', fontsize=18)
plt.ylabel('survey 1 scores', fontsize=18)
plt.scatter(['Male']*(len(male_s1_total)), male_s1_total, c=one, cmap='plasma_r', marker = ">")
one = np.arange(29)
plt.scatter(['Female']*(len(female_s1_total)), female_s1_total, c=one, cmap='plasma_r', marker = ">")
plt.show()

male_s2_total = []
female_s2_total = []
row = 0
for elt in data['Gender']:
    if elt == "Male":
        male_s2_total.append(data['S2_Total'][row])
    else:
        female_s2_total.append(data['S2_Total'][row])
    row += 1

# one-way ANOVA with gender against s2
gender_s2 = stats.f_oneway(male_s2_total, female_s2_total)
print(gender_s2)

one = np.arange(18)
plt.title('Gender vs. Self-Efficacy')
plt.xlabel('gender', fontsize=18)
plt.ylabel('survey 2 scores', fontsize=18)
plt.scatter(['Male']*(len(male_s2_total)), male_s2_total, c=one, cmap='plasma_r', marker = ">")
one = np.arange(29)
plt.scatter(['Female']*(len(female_s2_total)), female_s2_total, c=one, cmap='plasma_r', marker = ">")
plt.show()

# ethnicity
print("ethnicity stats")
asian_quiz1_total = []
white_quiz1_total = []
hispanic_quiz1_total = []
black_quiz1_total = []
mixed_quiz1_total = []
row = 0
for elt in data['Simple_Ethnicity']:
    if elt == "Asian":
        asian_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Hispanic":
        hispanic_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Black":
        black_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "White":
        white_quiz1_total.append(data['Q1_Total'][row])
    else:
        mixed_quiz1_total.append(data['Q1_Total'][row])
    row += 1

# one-way ANOVA with ethnicity against quiz1
ethnicity_quiz1 = stats.f_oneway(asian_quiz1_total, hispanic_quiz1_total, black_quiz1_total, mixed_quiz1_total, white_quiz1_total)
print(ethnicity_quiz1)

plt.title('Ethnicity vs. Performance')
plt.xlabel('ethnicities', fontsize=18)
plt.ylabel('quiz 1 scores', fontsize=18)
plt.scatter(['Asian']*(len(asian_quiz1_total)), asian_quiz1_total, c='cyan')
plt.scatter(['Hispanic']*(len(hispanic_quiz1_total)), hispanic_quiz1_total, c='magenta')
plt.scatter(['Black']*(len(black_quiz1_total)), black_quiz1_total, c='green')
plt.scatter(['Mixed']*(len(mixed_quiz1_total)), mixed_quiz1_total, c='yellow')
plt.scatter(['White']*(len(white_quiz1_total)), white_quiz1_total, c='blue')
plt.show()

asian_quiz2_total = []
hispanic_quiz2_total = []
black_quiz2_total = []
mixed_quiz2_total = []
white_quiz2_total = []
row = 0
for elt in data['Simple_Ethnicity']:
    if elt == "Asian":
        asian_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Hispanic":
        hispanic_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Black":
        black_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "White":
        white_quiz2_total.append(data['Q1_Total'][row])
    else:
        mixed_quiz2_total.append(data['Q2_Total'][row])
    row += 1

# one-way ANOVA with ethnicity against quiz2
ethnicity_quiz2 = stats.f_oneway(asian_quiz2_total, hispanic_quiz2_total, black_quiz2_total, mixed_quiz2_total, white_quiz2_total)
print(ethnicity_quiz2)

plt.title('Ethnicity vs. Performance')
plt.xlabel('ethnicities', fontsize=18)
plt.ylabel('quiz 2 scores', fontsize=18)
plt.scatter(['Asian']*(len(asian_quiz2_total)), asian_quiz2_total, c='cyan')
plt.scatter(['Hispanic']*(len(hispanic_quiz2_total)), hispanic_quiz2_total, c='magenta')
plt.scatter(['Black']*(len(black_quiz2_total)), black_quiz2_total, c='green')
plt.scatter(['Mixed']*(len(mixed_quiz2_total)), mixed_quiz2_total, c='yellow')
plt.scatter(['White']*(len(white_quiz2_total)), white_quiz2_total, c='blue')
plt.show()

asian_s1_total = []
hispanic_s1_total = []
black_s1_total = []
mixed_s1_total = []
white_s1_total = []
row = 0
for elt in data['Simple_Ethnicity']:
    if elt == "Asian":
        asian_s1_total.append(data['S1_Total'][row])
    elif elt == "Hispanic":
        hispanic_s1_total.append(data['S1_Total'][row])
    elif elt == "Black":
        black_s1_total.append(data['S1_Total'][row])
    elif elt == "White":
        white_s1_total.append(data['S1_Total'][row])
    else:
        mixed_s1_total.append(data['S1_Total'][row])
    row += 1

# one-way ANOVA with ethnicity against s1
ethnicity_s1 = stats.f_oneway(asian_s1_total, hispanic_s1_total, black_s1_total, mixed_s1_total, white_s1_total)
print(ethnicity_s1)

plt.title('Ethnicity vs. Self-Efficacy')
plt.xlabel('ethnicities', fontsize=18)
plt.ylabel('survey 1 scores', fontsize=18)
plt.scatter(['Asian']*(len(asian_s1_total)), asian_s1_total, c='cyan')
plt.scatter(['Hispanic']*(len(hispanic_s1_total)), hispanic_s1_total, c='magenta')
plt.scatter(['Black']*(len(black_s1_total)), black_s1_total, c='green')
plt.scatter(['Mixed']*(len(mixed_s1_total)), mixed_s1_total, c='yellow')
plt.scatter(['White']*(len(white_s1_total)), white_s1_total, c='blue')
plt.show()

asian_s2_total = []
hispanic_s2_total = []
black_s2_total = []
mixed_s2_total = []
white_s2_total = []
row = 0
for elt in data['Simple_Ethnicity']:
    if elt == "Asian":
        asian_s2_total.append(data['S2_Total'][row])
    elif elt == "Hispanic":
        hispanic_s2_total.append(data['S2_Total'][row])
    elif elt == "Black":
        black_s2_total.append(data['S2_Total'][row])
    elif elt == "White":
        white_s2_total.append(data['S2_Total'][row])
    else:
        mixed_s2_total.append(data['S2_Total'][row])
    row += 1

# one-way ANOVA with ethnicity against s2
ethnicity_s2 = stats.f_oneway(asian_s2_total, hispanic_s2_total, black_s2_total, mixed_s2_total, white_s2_total)
print(ethnicity_s2)

plt.title('Ethnicity vs. Self-Efficacy')
plt.xlabel('ethnicities', fontsize=18)
plt.ylabel('survey 2 scores', fontsize=18)
plt.scatter(['Asian']*(len(asian_s2_total)), asian_s2_total, c='cyan')
plt.scatter(['Hispanic']*(len(hispanic_s2_total)), hispanic_s2_total, c='magenta')
plt.scatter(['Black']*(len(black_s2_total)), black_s2_total, c='green')
plt.scatter(['Mixed']*(len(mixed_s2_total)), mixed_s2_total, c='yellow')
plt.scatter(['White']*(len(white_s2_total)), white_s2_total, c='blue')
plt.show()

# native language
print("native language stats")
english_quiz1_total = []
turkish_quiz1_total = []
bilingual_quiz1_total = []
chinese_quiz1_total = []
spanish_quiz1_total = []
portuguese_quiz1_total = []
tamil_quiz1_total = []
row = 0
for elt in data['Native_Lang']:
    if elt == "English":
        english_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Turkish":
        turkish_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Chinese":
        chinese_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Spanish":
        spanish_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Portuguese":
        portuguese_quiz1_total.append(data['Q1_Total'][row])
    elif elt == "Tamil":
        tamil_quiz1_total.append(data['Q1_Total'][row])
    else:
        bilingual_quiz1_total.append(data['Q1_Total'][row])
    row += 1

non_english_quiz1_total = turkish_quiz1_total + chinese_quiz1_total + spanish_quiz1_total + portuguese_quiz1_total + tamil_quiz1_total
# one-way ANOVA with native language against quiz1
language_quiz1 = stats.f_oneway(english_quiz1_total, non_english_quiz1_total, bilingual_quiz1_total)
print(language_quiz1)

plt.title('Native Language vs. Performance')
plt.xlabel('languages', fontsize=18)
plt.ylabel('quiz 1 scores', fontsize=18)
plt.scatter(['English']*(len(english_quiz1_total)), english_quiz1_total, c='cyan')
plt.scatter(['Non-English']*(len(non_english_quiz1_total)), non_english_quiz1_total, c='magenta')
plt.scatter(['Bilingual']*(len(bilingual_quiz1_total)), bilingual_quiz1_total, c='green')
plt.show()

english_quiz2_total = []
turkish_quiz2_total = []
bilingual_quiz2_total = []
chinese_quiz2_total = []
spanish_quiz2_total = []
portuguese_quiz2_total = []
tamil_quiz2_total = []
row = 0
for elt in data['Native_Lang']:
    if elt == "English":
        english_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Turkish":
        turkish_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Chinese":
        chinese_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Spanish":
        spanish_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Portuguese":
        portuguese_quiz2_total.append(data['Q2_Total'][row])
    elif elt == "Tamil":
        tamil_quiz2_total.append(data['Q2_Total'][row])
    else:
        bilingual_quiz2_total.append(data['Q2_Total'][row])
    row += 1

non_english_quiz2_total = turkish_quiz2_total + chinese_quiz2_total + spanish_quiz2_total + portuguese_quiz2_total + tamil_quiz2_total
# one-way ANOVA with native language against quiz2
language_quiz2 = stats.f_oneway(english_quiz2_total, non_english_quiz2_total, bilingual_quiz2_total)
print(language_quiz2)

plt.title('Native Language vs. Performance')
plt.xlabel('languages', fontsize=18)
plt.ylabel('quiz 2 scores', fontsize=18)
plt.scatter(['English']*(len(english_quiz2_total)), english_quiz2_total, c='cyan')
plt.scatter(['Non-English']*(len(non_english_quiz2_total)), non_english_quiz2_total, c='magenta')
plt.scatter(['Bilingual']*(len(bilingual_quiz2_total)), bilingual_quiz2_total, c='green')
plt.show()

english_s1_total = []
turkish_s1_total = []
bilingual_s1_total = []
chinese_s1_total = []
spanish_s1_total = []
portuguese_s1_total = []
tamil_s1_total = []
row = 0
for elt in data['Native_Lang']:
    if elt == "English":
        english_s1_total.append(data['S1_Total'][row])
    elif elt == "Turkish":
        turkish_s1_total.append(data['S1_Total'][row])
    elif elt == "Chinese":
        chinese_s1_total.append(data['S1_Total'][row])
    elif elt == "Spanish":
        spanish_s1_total.append(data['S1_Total'][row])
    elif elt == "Portuguese":
        portuguese_s1_total.append(data['S1_Total'][row])
    elif elt == "Tamil":
        tamil_s1_total.append(data['S1_Total'][row])
    else:
        bilingual_s1_total.append(data['S1_Total'][row])
    row += 1

non_english_s1_total = turkish_s1_total + chinese_s1_total + spanish_s1_total + portuguese_s1_total + tamil_s1_total
# one-way ANOVA with native language against s1
language_s1 = stats.f_oneway(english_s1_total, non_english_s1_total, bilingual_s1_total)
print(language_s1)

plt.title('Native Language vs. Self-Efficacy')
plt.xlabel('languages', fontsize=18)
plt.ylabel('survey 1 scores', fontsize=18)
plt.scatter(['English']*(len(english_s1_total)), english_s1_total, c='cyan')
plt.scatter(['Non-English']*(len(non_english_s1_total)), non_english_s1_total, c='magenta')
plt.scatter(['Bilingual']*(len(bilingual_s1_total)), bilingual_s1_total, c='green')
plt.show()

english_s2_total = []
turkish_s2_total = []
bilingual_s2_total = []
chinese_s2_total = []
spanish_s2_total = []
portuguese_s2_total = []
tamil_s2_total = []
row = 0
for elt in data['Native_Lang']:
    if elt == "English":
        english_s2_total.append(data['S2_Total'][row])
    elif elt == "Turkish":
        turkish_s2_total.append(data['S2_Total'][row])
    elif elt == "Chinese":
        chinese_s2_total.append(data['S2_Total'][row])
    elif elt == "Spanish":
        spanish_s2_total.append(data['S2_Total'][row])
    elif elt == "Portuguese":
        portuguese_s2_total.append(data['S2_Total'][row])
    elif elt == "Tamil":
        tamil_s2_total.append(data['S2_Total'][row])
    else:
        bilingual_s2_total.append(data['S2_Total'][row])
    row += 1

non_english_s2_total = turkish_s2_total + chinese_s2_total + spanish_s2_total + portuguese_s2_total + tamil_s2_total
# one-way ANOVA with native language against s2
language_s2 = stats.f_oneway(english_s2_total, non_english_s2_total, bilingual_s2_total)
print(language_s2)

plt.title('Native Language vs. Self-Efficacy')
plt.xlabel('languages', fontsize=18)
plt.ylabel('survey 2 scores', fontsize=18)
plt.scatter(['English']*(len(english_s2_total)), english_s2_total, c='cyan')
plt.scatter(['Non-English']*(len(non_english_s2_total)), non_english_s2_total, c='magenta')
plt.scatter(['Bilingual']*(len(bilingual_s2_total)), bilingual_s2_total, c='green')
plt.show()
