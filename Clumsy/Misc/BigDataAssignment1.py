from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.TalReader import TalReader
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.readers.IndexReader import JsonIndexReader


class Sleep():
    def __init__(self, subj='R1293P'):
        self.subj = subj
        self.exps = reader.experiments(subject=subj)
        self.main()

    def read_subj(self, exp):
        jr = JsonIndexReader('/protocols/r1.json')
        self.ev_paths = jr.aggregate_values('all_events', subject=self.subj, experiment=exp)
        self.evs = np.concatenate([BaseEventReader(filename=path).read() for path
                                   in sorted(self.ev_paths)]).view(np.recarray)

    def get_good_trodes(self):
        """
        The code below was created by Nora and will go through and read the text files
        good_leads and bad_leads in order to ensure that only good electrodes are read

        Parameters:

        Input:
            subjects: string ID of a subject (e.g. 'R1111M')

        Outputs:
            mpChans: identical to mp gotten other way, except excludes bad chs
            bpChans: identical to bp gotten other way, except excludes bad chs
        """

        # This looks wonky so it can easily be changed to run multiple subjects or a single subject as below
        talPath = []
        # Go through subjects, or here one subject and get tal paths
        for i, j in enumerate(self.subj):
            subject = self.subj
            tal_path = os.path.join('/data/eeg/', subject, 'tal', subject + '_talLocs_database_bipol.mat')
            talPath.append(tal_path)
        # get mp/bp channels
        monopolar_channels = [
            tal_readeri.get_monopolar_channels() for tal_readeri in [
                TalReader(filename=talPathi) for talPathi in talPath]]
        bipolar_pairs = [
            tal_readeri.get_bipolar_pairs() for tal_readeri in [
                TalReader(filename=talPathi) for talPathi in talPath]]

        # Exclude bad channels
        bpPairs = []
        mpChans = []
        countt = 0
        for talPathi, bipolar_pairsi, monopolar_channelsi in zip(
                talPath, bipolar_pairs, monopolar_channels):
            gf = open(os.path.dirname(talPathi) + '/good_leads.txt', 'r')
            goodleads = gf.read().split('\n')
            gf.close
            if os.path.isfile(os.path.dirname(talPathi) + '/bad_leads.txt'):
                bf = open(os.path.dirname(talPathi) + '/bad_leads.txt', 'r')
                badleads = bf.read().split('\n')
                bf.close
            else:
                badleads = []
            subPairs = np.array([pairi for pairi in bipolar_pairsi
                                 if (pairi[0].lstrip('0') in goodleads)
                                 and (pairi[1].lstrip('0') in goodleads)
                                 and (pairi[0].lstrip('0') not in badleads)
                                 and (pairi[1].lstrip('0') not in badleads)])
            subPairs = np.rec.array(subPairs)
            subChans = np.array([chani for chani in monopolar_channelsi
                                 if (chani.lstrip('0') in goodleads)
                                 and (chani.lstrip('0') not in badleads)])
            bpPairs.append(subPairs)
            mpChans.append(subChans)
        bp = bpPairs[0]  # wonky for reasons stated above
        mp = mpChans[0]
        self.bp = bp
        self.mp = mp
        tal_reader = TalReader(filename=tal_path)
        tal_structs = tal_reader.read()

        """Get only good pairs; int conversion necessary to convert [001,002]
        into [1,2] for comparasion match"""
        good_pairs = np.array([np.array([(int(x)) for i, x in enumerate(jj)]
                                        ) for ii, jj in enumerate(bp)])

        # Get indx of good tals
        channels_indx = []
        for i, ch in enumerate(tal_structs.channel):
            for x, pair in enumerate(good_pairs):
                if (pair[0] in ch) & (pair[1] in ch):
                    channels_indx.append(i)

        # Get good tals from all tals using index
        good_tal = np.array([tal for indx, tal in enumerate(tal_structs) if indx
                             in channels_indx]).view(np.recarray)
        # Get coords
        if len(good_tal) == len(bp):
            self.tal = good_tal
            if (('avgSurf' in good_tal.dtype.names) == True) and (
                        'x_snap' in good_tal.avgSurf.dtype.names) == True:
                self.x = good_tal.avgSurf.x_snap
                self.y = good_tal.avgSurf.y_snap
                self.z = good_tal.avgSurf.z_snap
            else:
                if ('x' in good_tal.dtype.names) == True:
                    print
                    'no avg tals sadface'
                    self.x, self.y, self.z = good_tal.x, good_tal.y, good_tal.z

    def get_eeg(self, events):
        # -------->Load data into EEG
        eeg_reader = EEGReader(events=events, channels=self.mp,
                               start_time=-2.0, end_time=2.0, buffer_time=1.0)
        self.eegs = eeg_reader.read()
        # -------->BP montage using good BP
        #
        m2b = MonopolarToBipolarMapper(time_series=self.eegs, bipolar_pairs=self.bp)
        bp_eegs = m2b.filter()
        """Remove 58-62 Hz"""
        if 'FR' not in self.subject:  # If not European remove 58-62
            b_filter = ButterworthFilter(time_series=bp_eegs, freq_range=[58., 62.], filt_type='stop', order=4)
            bp_eegs_filtered = b_filter.filter()
        if 'FR' in self.subject:  # If european remove 48-52
            b_filter = ButterworthFilter(time_series=bp_eegs, freq_range=[48., 52.], filt_type='stop', order=4)
            bp_eegs_filtered = b_filter.filter()
        self.bp_eegs = bp_eegs_filtered

    def Morlet(self, eeg):
        s = time()
        t = np.logspace(np.log10(2), np.log10(120), 35)[4:12]
        g = np.logspace(np.log10(2), np.log10(120), 35)[26:-1]
        freqs = np.concatenate((t, g))

        wf = MorletWaveletFilter(time_series=eeg,
                                 freqs=np.array(freqs), width=4,
                                 # output='power', #This can be either power, phase, or empty and retrun both
                                 # output: return np.abs(wavelet_coef_array) ** 2, None
                                 )
        pow_wavelet, phase_wavelet = wf.filter()

        # Remove the buffer
        pow_wavelet = pow_wavelet.remove_buffer(duration=1.0)
        np.log10(pow_wavelet.data, out=pow_wavelet.data);
        self.pow_wavelet = pow_wavelet
        phase_wavelet = phase_wavelet.remove_buffer(duration=1.0)
        self.phase_wavelet = phase_wavelet
        print(time() - s)

    def main(self):
        self.read_subj(exp=str(self.exps[-1]))
        # self.get_good_trodes()
        # self.get_eeg(self.evs)


from numba import jit


def load_all_events(MSD_instance):
    """
    Uses the MemorySearchDynamics Class to extract out all event arrays
    as well as a single subject's (LTP093 by default)

    ------
    INPUTS
    MSD_instance: an instance of MemorySearchDynamic class
    """
    # GET EVENTS
    all_events = MSD_instance.all_events
    events = all_events[all_events['subject'] == MSD_instance.subjs[0]]
    return all_events, events


def get_all_events_d(all_events):
    """
    Returns a dictionary where each key is a subject and each value is
    the event of that subject
    -------
    INPUTS
    all_events: an arr containing all events
    """
    d = {subj: all_events[all_events['subject'] == subj] for i, subj in enumerate(
        np.unique(all_events['subject']))}
    return d


def list_of_dict_to_dict(d_list):
    '''
    Take a list of dictionaries and turn it into one dictionary
    ------
    INPUTS:
    d_list: a list of dictionaries
    ------
    RETURNS:
    d: a dictionary that's a merge of the values of the dictionaries in d_list
    '''
    d = {}
    # Go through dicts, add values into d
    for dictionaries in d_list:
        for keys, values in enumerate(dictionaries):
            if values not in d:
                d[values] = []
            if values in d:
                d[values].append(dictionaries[values])
    return d


def list_to_df(df_lists):
    """
    Converts a list of DataFrames into one DataFrame,
    averaging across indexes
    -----
    INPUTS
    df_lists: a list of pd.DataFrames
    """
    concat = pd.concat(df_lists)
    concat = concat.groupby(concat.index).mean()
    return concat


def get_good_events(events, sess, trial):
    """
    Takes in events array and returns out one without any intrusions or
    any repeats. Repeats are removed entirely (both) same with intrusions.

    INPUTS
    ------
    events: an events np.recarray
    sess: The session to apply it over
    Trial: The trial to apply it over
    """
    # Get single sess/trial for subj, only rec words no intrusion
    events = events[(events['session'] == sess)
                    & (events['list'] == trial)
                    & (events['type'] == 'REC_WORD')
                    & (events['intrusion'] == 0)]
    d = dict(pd.Series(events['item_name']).value_counts())
    # Isolate bad words
    bad_words = [v for k, v in enumerate(d) if d[v] > 1]
    # Seperate out the bad words
    if len(bad_words) > 0:
        events = np.array([event for event in events
                           if event['item_name'] not in bad_words])  # .view(np.recarray)
    return events


def get_field(events, fields, sess, trial):
    """
    Gets the difference in the field of a single sess/trial of
    inputted events after filtering it using get_good_events

    INPUTS
    -----
    events: an array of beh events
    sess: session number must not be 0 or >23
    trial: number between 1-24
    """
    events = get_good_events(events, sess, trial)
    # Difference between sequential events
    if len(events) > 1:
        return events[fields]
    else:
        # print 'no events in sess {} trial {}'.format(sess,trial)
        return []


def get_field_diff(events, fields, sess, trial):
    """
    Gets the difference in the field of a single sess/trial of
    inputted events after filtering it using get_good_events

    INPUTS
    -----
    events: an array of beh events
    sess: session number must not be 0 or >23
    trial: number between 1-24
    """
    events = get_good_events(events, sess, trial)
    # Difference between sequential events
    if len(events) > 1:
        # Grab field
        diff = np.diff(events[fields])
        k = len(diff)
        if k > 0:
            return diff
    else:
        # print 'no events in sess {} trial {}'.format(sess,trial)
        return []


def get_field_diff_per_sess(events, sess, fieldname):
    """
    Gets all IRT per session of inputted events
    INPUTS
    -----
    events: an array of beh events
    sess: a session
    """
    # Get all lists from one sess
    session = [get_field_diff(events, fieldname, sess, i) for i in range(1, 25)]
    # Make the index be the len and then we'll group by that
    indx = [len(x) for x in session]
    try:
        df = pd.DataFrame(session, index=indx).groupby(indx).mean()
        return df
    except:
        pass
        # print indx


def get_field_per_subj(events, fieldname):
    """
    Gets ALL FIELDS fieldname of a single subject
    --------
    INPUTS
    events: events for a subject
    """
    all_irts = [get_field_diff_per_sess(events, i, fieldname
                                        ) for i in range(1, 24)]
    return list_to_df(all_irts)


def diff_in_field(events, fields, sess, trial, sp=None):
    """
    Gets the difference in the field of a single sess/trial of
    inputted events after filtering it using get_good_events,
    THIS VERSION WILL BE USED TO INTERACT WITH SERIALPOS SUCH THAT
    WE CAN CONTROL IF WE DO 0, 1, 2, ALL, ETC.

    INPUTS
    -----
    events: an array of beh events
    sess: session number must not be 0 or >23
    trial: number between 1-24
    """
    events = get_good_events(events, sess, trial)
    # Difference between sequential events
    if len(events) > 1:
        # Grab field
        diff = np.diff(events[fields])
        if type(sp) == int:
            try:
                diff = diff[sp]
                return diff
            except:
                return []
        elif type(sp) == str:
            if len(diff) > 0:
                return diff
    else:
        # print 'no events in sess {} trial {}'.format(sess,trial)
        return []


def plot_irt(irt_per_sess, title):
    """
    Plots an IRT curve, one line per len(recall)
    ------
    INPUTS
    irt_per_sess: output of get_irt_per_sess()
    """
    from matplotlib import pyplot as plt
    % matplotlib
    inline
    for i in range(len(irt_per_sess)):
        times = irt_per_sess.iloc[i].values
        pos = irt_per_sess.iloc[i].index
        # Remove all those pesky nans
        x = times[~np.isnan(times)]
        pos = pos[:(len(x))] + 1
        plt.plot(pos, x, label='len {}'.format(len(pos)), marker='*')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)
        plt.title('Inter-Response Interval ' + title)
        plt.ylabel('Time (ms)')
        plt.xlabel('Recall Serial Position')
    plt.show()


def get_irt_all(all_events_d):
    """
    Gets IRT for all subjects, generates a plot
    ------
    INPUTS
    all_events_d: a dictionary where keys=subj_id, and values=events_subj
    """
    my_list = [get_field_per_subj(all_events_d[subj], 'rectime'
                                  ) for subj in all_events_d.keys()]
    try:
        df = list_to_df(my_list)
        plot_irt(df)
        plt.show()
        return df
    except:
        try:
            return list_to_df(my_list)
        except:
            return my_list