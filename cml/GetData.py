# Script to use cmlreaders to easily get data on subjects
from cmlreaders import CMLReader, get_data_index

def get_subjects(experiment):
    """

    Parameters
    ----------
    experiment

    Returns
    -------

    """
    if experiment.lower() in ('catfr1', 'fr1'):
        dataframe = get_data_index("r1")

    elif experiment.lower() in ('pyfr'):
        dataframe = get_data_index("pyfr")

    elif experiment.lower() in ('ltp', 'ltpfr2'):
        dataframe = get_data_index("ltp")
    else:

        try:
            dataframe = get_data_index(experiment)
        except ValueError as e:
            print(e)
            return

    return dataframe[dataframe['experiment'] == experiment]['subject'].unique()


def get_sessions(subject, experiment):
    """

    Parameters
    ----------
    subject
    experiment

    Returns
    -------

    """
    RAM = ['catFR1', 'FR1']
    if experiment in RAM:
        dataframe = get_data_index("r1")
        return dataframe[(dataframe['experiment'] == experiment)
                         & (dataframe['subject'] == subject)]['session'].unique()
    elif experiment == 'pyFR':
        dataframe = get_data_index("pyfr")
        return dataframe[(dataframe['experiment'] == experiment)
                         & (dataframe['subject'] == subject)]['session'].unique()

def get_ram_experiments(subject):
    dataframe = get_data_index("r1")
    return dataframe[dataframe['subject']==subject]['experiment'].unique()