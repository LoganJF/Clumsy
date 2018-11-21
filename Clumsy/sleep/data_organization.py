from cmlreaders import CMLReader, get_data_index
from ptsa.data.readers import JsonIndexReader, TalReader
import os
import datetime
import numpy as np
import pandas as pd
__all__ = ['get_sleep_behavior_dataframe',
           'DATA_ROOTS',
           'SUBJ_EXP_SESS_DATE_MATCHER',
           'check_int_ram_formatting',
           'subject_from_path',
           'get_night_dates_before_and_after',
           'get_sleep_path',]

# Describes the current /data/eeg setup for all subjects
DATA_ROOTS = {
    'LFSleep' : {

        'ProcData': {

        }
        , 'RawData': {

        }
    }
}

# Describe relationships between experiments and when they took place
SUBJ_EXP_SESS_DATE_MATCHER = {

    'R1207J': {

        'catfr1': {
            '0':'08_17_16_1543_1610',
            '1':'8_18_16_1530_1610'
        },

        'fr1':{
            '0':'8_20_16_1021_1105',
        }
    },

    'R1230J': {

        'fr1': {
            '0':'10_09_16_1400_1500',
            '2':'10_12_16_1138_1227',
            '4':'10_13_16_1520_1615',
            '5':'10_14_16_1017_1114',
            '6':'10_17_16_1320_1409',
        },

        'catfr1': {
            '0': '10_10_16_1220_1307',
        }
    },

    'R1293P': {

        'catfr1': {
            '0': '04_17_17_0955_1055',
        },

        'fr1' : {
            '1' : '04_18_17_1625_1715',
        },
    },

    'R1348J' : {

        'catfr1': {
          '0' : '09_26_17_1540_1630'
        },

        'dboy3': {
            '0': '09_23_17_1506',
            '1': '09_26_17_1449',
            '2': '09_27_17_1533',
        },

    },

    'R1384J' : {

        'catfr1': {
          '0' : '09_26_17_1540_1630'
        },

        'dboy3': { # HOW SHOULD WE HANDLE??
            '0':'02_01_18_1825',
            '1':'02_03_18_1810',
            '2':'02_03_18_1812',
            '3':'02_07_18_1732',
            '4':'02_07_18_1734',
            '5':'02_10_18_1827',
            '6':'02_10_18_1828',
        },

        'dboy1' : {
            '1': '02_17_18',
        },
    },
}
# SHOULD ALL OF THE ABOVE BE IN A DIFFERENT FILE?
def check_int_ram_formatting(s):
    """Check to see if inputted string matches RAM format

    Utility function used in subject_from_path
    Parameters
    ----------
    s: str, various dir names

    Returns
    -------

    """
    if len(s) == 0: # Nothing passed from split?
        return False
    # Check if it matches RAM formatting
    if s[0] in ('R') and s[-1] in ('J', 'P'): # Jefferson, Penn. E.g. R1207J
        return s[1:-1].isdigit()
    return s.isdigit()

def subject_from_path(path):
    """Given a path will extract out the subject ID if it is formatted as RAM structure

    Parameters
    ----------
    path: str, path of relevant data structure containing a subject folder

    Returns
    -------
    subject ID: str
    """
    for split in path.split('/'):
        valid = check_int_ram_formatting(split)
        if valid:
            return split


def get_night_dates_before_and_after(subject, experiment = None, session = None):
    """Returns the relevant dates for a subject given either their experiment and session or the date and time

    Parameters
    ----------
    subject: str,
             subject ID, e.g. 'R1111M'
    experiment: str, by default None
            experiment tag, valid: 'FR1', 'catFR1', dboy1, dboy3
    session: int, by default None
            session number for the experiment

    Returns
    -------
    before: datetime.datetime object of the day before
    after: datetime.datetime object of the day after
    """
    global SUBJ_EXP_SESS_DATE_MATCHER
    s = SUBJ_EXP_SESS_DATE_MATCHER[subject][experiment][session]
    a = s.split('_')
    # Convert the literal path into int and check formatting
    month = int(a[0])
    day = int(a[1])
    year = int(a[2])
    if year < 2000: # E.g. 8_12_18 -> 8_12_2018
        year += 2000
    date = datetime.datetime(year=year, month=month, day=day)
    before = date - datetime.timedelta(days=1)
    after = date + datetime.timedelta(days=1)
    return before, after


def get_sleep_path(subject, experiment = None, session = None,  datetimes = None, resampled=100):
    """Returns the relevant sleep path for a subject given either their experiment and session or the date and time

    Parameters
    ----------
    subject: str,
             subject ID, e.g. 'R1111M'
    experiment: str, by default None
            experiment tag, valid: 'FR1', 'catFR1', dBoy
    session: int, by default None
            session number for the experiment
    datetimes: list, by default None
               two datetime.datetime objects consisting of the start and end point of the desired time
    resampled: int, by default 100 Hz
               for the given path collected, if there's a resampled version, then return that
               valid: 100, None
    Returns
    -------

    """
    # Check if we're using this locally or through rhino directly
    cwd = os.getcwd()
    local = '' if (cwd.split('/')[1][:4] == 'home') else '/Volumes/rhino'
    protocol = local + '/protocols/r1.json'
    global SUBJ_EXP_SESS_DATE_MATCHER
    return local + SUBJ_EXP_SESS_DATE_MATCHER[subject][experiment][session]


def _get_sleep_path(subject, experiment=None, session=None, datetimes=None):
    """Returns the relevant sleep path for a subject given either their experiment and session or the date and time

    Parameters
    ----------
    subject: str,
             subject ID, e.g. 'R1111M'
    experiment: str, by default None
            experiment tag, valid: 'FR1', 'catFR1', dBoy
    session: int, by default None
            session number for the experiment
    datetimes: list, by default None
               two datetime.datetime objects consisting of the start and end point of the desired time

    Returns
    -------

    """
    # Check if we're using this locally or through rhino directly
    # cwd = os.getcwd()
    # local = '' if (cwd.split('/')[1][:4] == 'home') else '/Volumes/rhino'
    # protocol = local + '/protocols/r1.json'
    global SUBJ_EXP_SESS_DATE_MATCHER
    return SUBJ_EXP_SESS_DATE_MATCHER[subject][experiment][session]

#subjects = ["R1207J", "R1230J", 'R1293P', "R1384J", "R1348J", "R1398J"]

def get_sleep_behavior_dataframe(subjects):
    """Return the relevant dates, experiments, session and performance for inputted list of subjects
    subjects: np.array like, list of subjects to analyze
    """
    subjs_df = CMLReader.get_data_index()
    subj_d = {subject : subjs_df[subjs_df['subject']==subject][['experiment', 'session']] for subject in subjects}
    for k,v in enumerate(subj_d):
        subj_d[v] = subj_d[v][(subj_d[v]['experiment']=='FR1' )| (subj_d[v]['experiment']=='catFR1')]

    for k,v in enumerate(subj_d):
        performances = []
        dates = []
        for exp, sess in  np.array(subj_d[v]):
            rdr = CMLReader(subject=v, experiment=exp, session=sess)
            events = rdr.load('events')
            misses, hits = events[events['type']=='WORD']['recalled'].value_counts()
            performance = hits/(misses+hits)*100
            performances.append(performance)
            _,_,_,date,_ = events['eegfile'].iloc[11].split('_')
            dates.append(date)
        subj_d[v]['subject'] = v
        subj_d[v]['performance'] = performances
        subj_d[v]['date'] = dates

    subj_df = pd.concat(subj_d.values())
