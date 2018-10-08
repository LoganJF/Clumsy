# Script to use cmlreaders to easily get data on subjects
from cmlreaders import CMLReader, get_data_index
from ptsa.data.readers import JsonIndexReader, TalReader
import os

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
    """

    Parameters
    ----------
    subject

    Returns
    -------

    """
    dataframe = get_data_index("r1")
    return dataframe[dataframe['subject']==subject]['experiment'].unique()


def get_sub_tal(subject, experiment, return_channels=False):
    """Returns a subject's talairach using TalReader and JsonIndexReader
    -----
    INPUTS:
    -----
    subject: str, subject ID, e.g. 'R1111M'
    experiment: str, experiment, e.g. 'FR1', 'catFR1'
    return_channels: bool, default = False, whether to return arrays of
                     monopolar and bipolar channels used for EEGReader
    ------
    OUTPUTS if return_channels is False:
    ------
    tal_reader.read(): np.recarray, an array containing relevant values
                       for electrode localization
    ------
    OUTPUTS if return_channels is True:
    ------
    mp: np.recarray, monopolar channels used for EEGReader
    bp: np.recarray, bipolar channels used for EEGReader
    tal_reader.read(): np.recarray, an array containing relevant values
                       for electrode localization
    """

    # Check if we're using this locally or through rhino directly
    cwd = os.getcwd()
    local = '' if (cwd.split('/')[1][:4] == 'home') else '/Volumes/rhino'
    protocol = local + '/protocols/r1.json'

    # Load the protocol
    jr = JsonIndexReader(protocol)

    # Get the path for the TalReader
    pairs_path = jr.get_value('pairs', subject=subject, experiment=experiment)

    # Create an instance of the TalReader object
    tal_reader = TalReader(filename=pairs_path)

    # If desired, also return the monopolar and bipolar channels
    if return_channels:
        mp = tal_reader.get_monopolar_channels()
        bp = tal_reader.get_bipolar_pairs()
        return mp, bp, tal_reader.read()

    # Otherwise just return the tal
    return tal_reader.read()

if __name__ == '__main__':
    mp, bp, bp_tal = get_sub_tal(subject='R1207J', experiment='FR1', return_channels=True)