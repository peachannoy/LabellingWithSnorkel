import snorkel

import glob
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function


filenames=sorted(glob.glob('B:\\Desktop\\Studium Gruppenarbeiten\\LabellingWithSnorkel\\weatherAUS.csv'))
print( filenames)

from Datencorrelation import load_spam_dataset

df_train, df_test = load_spam_dataset()

Y_test = df_test.label.values

# For clarity, we define constants to represent the class labels for spam, ham, and abstaining.
#ABSTAIN = -1
NORAIN = 0
RAIN = 1


@labeling_function()
def rainToday(x):
    return RAIN if x==1 else NORAIN

@labeling_function()
def humidity3pm(x):
    return RAIN if x>50 else NORAIN

@labeling_function
def cloud3pm(x):
    return RAIN if x>4 else NORAIN

@labeling_function
def windGustSpeed(x):
    return RAIN if x>30 else NORAIN

@labeling_function
def sunshine(x):
    return RAIN if x<5 else NORAIN

@labeling_function
def pressure9am(x):
    return RAIN if x<1005 else NORAIN

lfs = [
    rainToday,
    humidity3pm,
    cloud3pm,
    windGustSpeed,
    sunshine,
    pressure9am

]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)