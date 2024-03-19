# -*- coding: utf-8 -*-
import librosa
import numpy as np
from musicxml import *
from .tabCNN import model
import os
from datetime import datetime

class guitar2Tab:
    def __init__ (self,
                  sr=22050,
                  hop_length=256,
                 ):
        
        self.sr = sr
        self.hop_length = hop_length

    # assign value to its closest neighbour
    def quantize(self, val, to_values:list=[]):
        """Quantize a value with regards to a set of allowed values.

        Examples:
            quantize(49.513, [0, 45, 90]) -> 45
            quantize(43, [0, 10, 20, 30]) -> 30

        Note: function doesn't assume to_values to be sorted and
        iterates over all values (i.e. is rather slow).

        Args:
            val        The value to quantize
            to_values  The allowed values
        Returns:
            Closest value among allowed values.
        """

        if(len(to_values) == 0):
            return

        note_type = {
            self.half: 'half',
            self.whole: 'whole',
            self.quarter: 'quarter',
            self.eighth: 'eighth',
            self.sixteenth: '16th',
        }

        best_match = None
        best_match_diff = None
        for other_val in to_values:
            diff = abs(other_val - val)
            if best_match is None or diff < best_match_diff:
                best_match = other_val
                best_match_diff = diff
        return note_type[best_match]

    # convert midi value to note
    def number_to_note(self, number: int) -> tuple:
        # convert midi to note
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        OCTAVES = list(range(11))
        NOTES_IN_OCTAVE = len(NOTES)

        octave = number // NOTES_IN_OCTAVE
        note = NOTES[number % NOTES_IN_OCTAVE]

        return note, octave

    # convert prediction to tab
    def tab2bin(self, tab):
        tab_arr = np.zeros((6,20))
        for string_num in range(len(tab)):
            fret_vector = tab[string_num]
            fret_class = np.argmax(fret_vector, -1)
            # 0 means that the string is closed
            if fret_class > 0:
                fret_num = fret_class - 1
                tab_arr[string_num][fret_num] = 1
        return tab_arr

    # convert predictions to pitch
    def tab2pitch(self, tab):
        pitch_vector = np.zeros(44)
        string_pitches = [40, 45, 50, 55, 59, 64]
        for string_num in range(len(tab)):
            fret_vector = tab[string_num]
            fret_class = np.argmax(fret_vector, -1)
            # 0 means that the string is closed
            if fret_class > 0:
                # 21 + 64 - 41 = 44 is the max number
                pitch_num = fret_class + string_pitches[string_num] - 41
                pitch_vector[pitch_num] = 1
        return pitch_vector

    # CQT
    ## Function
    def calc_cqt(self, x,fs=22050,hop_length=256, n_bins=192, cqt_bins_per_octave=24, mag_exp=4):
        new_x = x.astype(float)
        new_x = librosa.util.normalize(new_x)
        C = np.abs(librosa.cqt(new_x,
            hop_length=hop_length,
            sr=fs,
            n_bins=n_bins,
            bins_per_octave=cqt_bins_per_octave))

        return C

    # CQT Threshold
    def cqt_thresholded(self, cqt,thres=-80):
        C_mag = librosa.magphase(cqt)[0]**4
        CdB = librosa.amplitude_to_db(C_mag ,ref=np.max)
        low_mag_indices = np.where(CdB < thres)
        new_cqt=np.copy(cqt)
        new_cqt[low_mag_indices] = 0.01
        return new_cqt

    # Onset Envelope from Cqt
    def calc_onset_env(self, y, sr=22050, hop_length=256):
        return librosa.onset.onset_strength(S=y, sr=sr)

    # Onset from Onset Envelope
    def calc_onset(self, y, sr=22050, hop_length=256, pre_post_max=6, backtrack=True):
        onset_env=self.calc_onset_env(y)
        onset_frames = librosa.onset.onset_detect(
                                            y=None,
                                            onset_envelope=onset_env,
                                            sr=sr,
                                            units='frames',
                                            hop_length=hop_length,
                                            backtrack=backtrack,
                                            wait=0.002, pre_avg=0.002, post_avg=0.002, pre_max=0.002, post_max=0.002,
                                            )
        onset_boundaries = np.concatenate([[0], onset_frames])
        onset_times = librosa.frames_to_time(onset_boundaries, sr=sr, hop_length=hop_length)
        return [onset_times, onset_boundaries, onset_env]
    
    def extract_key_signature(self, y, sr):
        # Compute the Chroma Short-Time Fourier Transform (chroma_stft)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

        # Calculate the mean chroma feature across time
        mean_chroma = np.mean(chromagram, axis=1)

        # Define the mapping of chroma features to keys
        chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Find the key by selecting the maximum chroma feature
        estimated_key_index = np.argmax(mean_chroma)
        estimated_key = chroma_to_key[estimated_key_index]

        key = estimated_key.split('#')[0]

        # key_mapping = {
        #     'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
        #     'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7,
        # }
        key_alter = 1
        # fifths = key_mapping[key[0]]
        # Determine the number of sharps or flats
        if "B" in estimated_key:
            key_alter = -1  # Sharp

        return {
            "key": key,
            "key_alter": key_alter,
            "full_key": estimated_key
        }



    # convert prediction to tab and its notes
    def extract_note_info(self, tab):
        out1 = self.tab2bin(tab)

        # if(len(np.where(out == 1)[0]) > 0):
        if(len(np.where(out1 == 1)[0]) > 0):
            # pitch = tab2pitch(tab)
            fret_string_pos = np.where(out1 == 1)

            pitch_list = list(range(40, 77))
            pitches_from_tab = np.where(self.tab2pitch(tab) == 1)

            midi_notes = [pitch_list[x_pitch] for x_pitch in pitches_from_tab[0]]
            midi_notes = [self.number_to_note(midi_note) for midi_note in midi_notes]

            return {
                "frets_strings": fret_string_pos,
                "midi": midi_notes
            }
        else:
            return "none"
        

    def predict(self, file_audio, filename:str):
        y, sr = librosa.load(file_audio, sr=22050)

        # sr = 22050

        # STFT parameters
        hop_length = 512

        # reassign y audio to y3
        y3 = y

        # onset settings
        pre_post_max=10
        backtrack=True

        # preprocess audio by minimising less apparent noises in the sample
        C = self.calc_cqt(x=y, hop_length=hop_length)
        new_cqt=self.cqt_thresholded(C,-80)
        o1 = new_cqt    

        # swap the dimension
        x1 = np.swapaxes(o1, 0, 1)

        # calculate onsets
        onsets=self.calc_onset(new_cqt, hop_length=hop_length, pre_post_max=pre_post_max, backtrack=backtrack)

        o_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(C, ref=np.max), sr=22050)
        onset_raw = librosa.onset.onset_detect(y=y, onset_envelope=o_env, sr=22050, hop_length=hop_length, wait=0.2, pre_avg=0.2, post_avg=0.2, pre_max=0.2, post_max=0.2)
        # onsets1 = librosa.frames_to_time(onsets[1], sr=22050, hop_length=hop_length)

        S = np.abs(librosa.stft(y=y, hop_length=hop_length))
        rms = librosa.feature.rms(S=S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])
        onsets1 = librosa.frames_to_time(onset_bt_rms)

        print(
            onsets1,
            onset_raw,
        )

        # Estimate Tempo
        tempo, beats = librosa.beat.beat_track(y=None, sr=sr, onset_envelope=onsets1, hop_length=hop_length,
                    start_bpm=90.0, tightness=100, trim=True, bpm=None,
                    units='frames')
        
        tempo=int(2*round(tempo/2))

        # Estimate Key Signature
        key_signature = self.extract_key_signature(y, sr)

        # calculate beat type in seconds for quantization
        self.quarter = 60/tempo
        self.half = self.quarter * 2
        self.whole = self.half * 2
        self.eighth = self.quarter / 2
        self.sixteenth = self.eighth / 2
        self.th32 = self.sixteenth / 2
        self.th64 = self.th32 / 2

        # load in pre-trained weights
        new_model = model(weights=f"{os.getcwd()}/model/weights.h5")
        predictor = new_model.build_model()

        # predictor.load_weights("weights.h5")

        # hot swap dimensions of the matrix
        dur = x1.shape[0]
        halfwin = 9 // 2
        label_dim = (6,21)
        X_dim = (dur, 192, 9, 1)
        y_dim = (192, label_dim[0], label_dim[1])
        X = np.empty(X_dim)
        y = np.empty(y_dim)

        # add con window between each time frame
        for i in range(dur):
            full_x = np.pad(x1, [(halfwin,halfwin), (0,0)], mode='constant')
            sample_x = full_x[i : i + 9]
            X[i,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)

        # predict audio
        prediction = predictor.predict(X)

        # temp = []
        # pos = []
        # # get the indices of the time frame that has note(s) in it
        # for x in range(dur):
        #     out = self.tab2bin(prediction[x])
        #     if(len(np.where(out == 1)[0]) > 0):
        #         # if(x in beat_frames):
        #         temp.append(x)
        #         pos.append(x)

        def map_onsets_to_results(onsets, results):
            mapped_results = []
            for onset_frame in onsets:
                pred_frame = self.tab2bin(results[onset_frame])
                if (len(np.where(pred_frame == 1)[0]) > 0):
                    mapped_results.append({ **self.extract_note_info(results[onset_frame]), 'beat_type': 'eighth' })
                else:
                    for y in range(max(onset_frame - 5, 0), min(onset_frame + 5, len(results) - 1)):
                        pred_frame = self.tab2bin(results[y])
                        if (len(np.where(pred_frame == 1)[0]) > 0):
                            mapped_results.append({ **self.extract_note_info(results[y]), 'beat_type': 'eighth' })
                            break
                    # closest_frame = np.argmin(np.abs(results - onset_frame))
                    # pred_closest_frame = self.tab2bin(results[closest_frame])
                    # if (len(np.where(pred_closest_frame == 1)[0]) > 0):
                    #     mapped_results.append({ **self.extract_note_info(results[closest_frame]), 'beat_type': 'eighth' })
                    # else:
                    #     mapped_results.append({ **self.extract_note_info(results[closest_frame]), 'beat_type': 'eighth' })
            return mapped_results
        
        print(x1.shape)
        
        # print(map_onsets_to_results(onsets[1], prediction))

        # tmp = []
        results = map_onsets_to_results(onset_bt_rms, prediction)
        # for x in range(dur):
        #     out = self.tab2bin(prediction[x])
        #     if(len(np.where(out == 1)[0]) > 0):
        #         results.append({ **self.extract_note_info(prediction[x]), 'beat_type': 'eighth' })
            # if(x in onsets0):
                # tmp.append(self.extract_note_info(prediction[x]))
                # x = round(x / ratio)
                # idx = onsets0.index(x)
                # idx = np.where(onsets0 == x)[0][0]
                # next_dur = float(onsets[0][idx + 1]) if idx+1 < len(onsets[0]) else float(onsets[0][idx])

                # if(len(np.where(out == 1)[0]) > 0):
                #     results.append({ **self.extract_note_info(prediction[x]), 'beat_type': self.quantize((next_dur - float(onsets[0][idx])), [self.sixteenth, self.eighth, self.quarter, self.half, self.whole]) })
                # elif (x + 1 in pos):
                #     results.append({ **self.extract_note_info(prediction[x+1]), 'beat_type': self.quantize((next_dur - float(onsets[0][idx])), [self.sixteenth, self.eighth, self.quarter, self.half, self.whole]) })
                # else:
                #     results.append(self.extract_note_info(prediction[x]))

        # part-wise
        s1 = XMLScorePartwise()

        # part-list
        partwise = s1.add_child(XMLPartList())

        # score part
        gp = partwise.add_child(XMLScorePart(id="P1"))

        # part name and abbr
        gp.add_child(XMLPartName("Guitar", print_object="no"))
        gp.add_child(XMLPartAbbreviation("Gtr.", print_object="no"))

        # part
        p1 = s1.add_child(XMLPart(id="P1"))

        # invert string indices
        translate_string = {
            1: 6,
            2: 5,
            3: 4,
            4: 3,
            5: 2,
            6: 1
        }

        def add_measure(idx='1', notes=[]):
            # add measure
            m1 = p1.add_child(XMLMeasure(number=idx))

            # print
            m1_print = m1.add_child(XMLPrint())

            # measure numbering
            m1_print.add_child(XMLMeasureNumbering("none"))

            # attribute
            attr1 = m1.add_child(XMLAttributes())

            # add division
            attr1.add_child(XMLDivisions(480))

            # add attributes to the first measure
            if(idx == '1'):
                # add key
                k1 = attr1.add_child(XMLKey(print_object="yes"))
                k1.add_child(XMLKeyStep(str(key_signature['key'])))
                k1.add_child(XMLKeyAlter(int(key_signature['key_alter'])))

                # add time
                t1 = attr1.add_child(XMLTime(print_object="no"))
                t1.add_child(XMLBeats("4"))
                t1.add_child(XMLBeatType("4"))

                # add clef
                clef1 = attr1.add_child(XMLClef())
                clef1.add_child(XMLSign("TAB"))
                clef1.add_child(XMLLine(5))

                # add staffs
                std1 = attr1.add_child(XMLStaffDetails())
                std1.add_child(XMLStaffLines(6))

                # staff 1
                st1 = std1.add_child(XMLStaffTuning(line=1))
                st1.add_child(XMLTuningStep("E"))
                st1.add_child(XMLTuningOctave(2))

                # staff 2
                st2 = std1.add_child(XMLStaffTuning(line=2))
                st2.add_child(XMLTuningStep("A"))
                st2.add_child(XMLTuningOctave(2))

                # staff 3
                st3 = std1.add_child(XMLStaffTuning(line=3))
                st3.add_child(XMLTuningStep("D"))
                st3.add_child(XMLTuningOctave(3))

                # staff 4
                st4 = std1.add_child(XMLStaffTuning(line=4))
                st4.add_child(XMLTuningStep("G"))
                st4.add_child(XMLTuningOctave(3))

                # staff 5
                st5 = std1.add_child(XMLStaffTuning(line=5))
                st5.add_child(XMLTuningStep("B"))
                st5.add_child(XMLTuningOctave(3))

                # staff 6
                st6 = std1.add_child(XMLStaffTuning(line=6))
                st6.add_child(XMLTuningStep("E"))
                st6.add_child(XMLTuningOctave(4))

                # add sound/tempo
                d1 = m1.add_child(XMLDirection(placement='above', directive='yes'))
                d1t = d1.add_child(XMLDirectionType())

                # direction-type
                mtr = d1t.add_child(XMLMetronome())
                mtr.add_child(XMLBeatUnit('quarter'))
                mtr.add_child(XMLPerMinute(str(tempo)))

            # add notes to musicxml
            for note in notes:
                if(type(note) is dict):
                    add_note(measure=m1, pitch=note['midi'], dur=1, string_fret=note['frets_strings'], beat_type=note['beat_type'])
                else:
                    m1n1 = m1.add_child(XMLNote())
                    m1n1.add_child(XMLRest())
                    m1n1.xml_duration = dur
                    m1n1.xml_voice = '1'

        # add note
        def add_note(measure, dur=1, string_fret=[], pitch=[], beat_type:str = 'quarter'):
            for idx in range(len(pitch)):
                m1n1 = measure.add_child(XMLNote())

                if(len(string_fret[0]) > 1 and idx > 0):
                    # add note as part of a chord
                    m1n1.add_child(XMLChord())

                # add pitch to note
                m1p1 = m1n1.add_child(XMLPitch())

                if(len(pitch[idx]) > 1 and len(pitch[idx][0]) > 1):
                    m1p1.add_child(XMLStep(pitch[idx][0][0]))
                    m1p1.xml_alter = 1
                else:
                    m1p1.add_child(XMLStep(pitch[idx][0]))

                m1p1.xml_octave = pitch[idx][1]

                m1n1.add_child(XMLType('eighth'))
                m1n1.xml_duration = 30
                m1not1 = m1n1.add_child(XMLNotations())
                m1not1tech1 = m1not1.add_child(XMLTechnical())
                # if it is a chord
                if(len(string_fret[0]) > 1):
                    m1not1tech1.xml_string = translate_string[int(string_fret[0][idx]) + 1]
                    m1not1tech1.xml_fret = int(string_fret[1][idx])
                # if just a note
                else:
                    m1not1tech1.xml_string = translate_string[int(string_fret[0][0]) + 1]
                    m1not1tech1.xml_fret = int(string_fret[1][0])

        # add notes to each measure in step of 4
        measure = 0
        for itr in range(0, len(results), 4):
            measure += 1
            add_measure(idx=str(measure), notes=results[itr:itr+4])

        # save musicxml object to file path
        xml_path = f"output/{filename}.xml"
        s1.write(xml_path)

        return { 
                "filename": filename, 
                "bpm": tempo,
                "key": key_signature['full_key'],
                "time_signature": "4/4",
                "duration": librosa.get_duration(y=y3, sr=sr),
                "tuning": "Guitar Standard Tuning",
                "genre": "Unknown",
                }