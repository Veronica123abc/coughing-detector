
import pandas as pd
import os
import train
def imagimob_to_audacity(source_filename):
    imagimob_df = pd.read_csv(source_filename)
    imagimob_df['stop'] = imagimob_df['Time(Seconds)'] + imagimob_df['Length(Seconds)']
    audacity_df = {'start': imagimob_df['Time(Seconds)'],
                   'stop': imagimob_df['Time(Seconds)'] + imagimob_df['Length(Seconds)'],
                   'label':imagimob_df['Label(string)']}
    pd.DataFrame(audacity_df).to_csv('data/label_audacity.txt', index=False, header=False, sep='\t')
    return pd.DataFrame(audacity_df)
    #tmp.to_csv('data/label_audacity.txt', index=False, header=False, sep='\t')

def create_audacity_annotation_files(sub_dirs):
    backgrounds = []
    for sub_dir in sub_dirs:
        for root, dirs, files in os.walk(sub_dir):
            path = root.split(os.sep)
            if 'label.label' in files:
                print(path, ' ', files) #audacity_df = imagimob_to_audacity(os.)
                df_audacity = imagimob_to_audacity(os.path.join(root, 'label.label'))
                df_audacity.to_csv(os.path.join(root,'audacity_label.txt'), index=False, header=False, sep='\t')

def extract_labeled_files(root_dir):
    labeled_files = []
    for root, dirs, files in os.walk(root_dir):
        if 'data.wav' in files and 'label.label' in files:
            labeled_files.append(root)
    return labeled_files

def extract_background_files(sub_dirs):
    backgrounds = []
    for sub_dir in sub_dirs:
        for root, dirs, files in os.walk(sub_dir):
            path = root.split(os.sep)
            if 'data.wav' in files:
                backgrounds.append(os.path.join(root, 'data.wav'))
    return backgrounds


# def main():
#     # train_set = dataset.Coughing("training.txt")
#     # test_set = dataset.Coughing("test.txt")
#     #featuring.generate_train_test_validate('data/coughing', 'training.txt')
#     #create_audacity_annotation_files(['data/coughing_batch_2'])
#     # train.train()
#     # utils.inference('data/coughing/148310__elgeorgia__cough/data.wav')
#     # featuring.create_mel_specs_for_training('kallekula3')
#     # utils.plot_samples(annotations='training_2.txt', shuffle=False)
#     #utils.plot_samples(annotations='training.txt', shuffle=False)
#     #ms = featuring.mel_spectrogram(event)
#
#     # audio, sr =  synthesizer.merge([audio1, audio2], [0.5, 0.5])
#     # sd.play(audio, samplerate=sr)
#    # imagimob_to_audacity(label)

if __name__ == "__main__":
    main()
