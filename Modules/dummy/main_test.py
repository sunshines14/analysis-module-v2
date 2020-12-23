import os

#from AnalysisModule import settings
#from Modules.dummy.example import test
#from Modules.dummy.sed import sed
#from WebAnalyzer.utils.media import frames_to_timecode

class Dummy:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        #import keras
        #model_path = os.path.join(self.path, "model.h5")
        #self.model = keras.models.load_model(model_path)
        pass
        
    def inference_by_image(self, image_path):
        result = []
        # TODO
        #   - Inference using image path

        # result sample
        result = {"frame_result": [
            {
                # 1 bbox & multiple object
                'label': [
                    {'description': 'person', 'score': 1.0},
                    {'description': 'chair', 'score': 1.0}
                ],
                'position': {
                    'x': 0.0,
                    'y': 0.0,
                    'w': 0.0,
                    'h': 0.0
                }
            },
            {
                # 1 bbox & 1 object
                'label': [
                    {'description': 'car', 'score': 1.0},
                ],
                'position': {
                    'x': 100.0,
                    'y': 100.0,
                    'w': 100.0,
                    'h': 100.0
                }
            }
        ]}
        self.result = result

        return self.result

    def inference_by_video(frame_path_list, infos):
        results = []
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['extract_fps']
        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            result = self.inference_by_image(frame_path)
            result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
            result["frame_number"] = int((idx + 1) * fps)
            result["timestamp"] = frames_to_timecode((idx + 1) * fps, fps)
            results.append(result)

        self.result = results

        return self.result

    def inference_by_audio(audio_path):
        #video_info = infos['video_info']
        result_list = []
        result_dict = {}
        # TODO
        #   - Inference using image path
        #   -
        import keras
        import subprocess
        from sed import sed

        model = keras.models.load_model('model.h5')
        subprocess.call(['/workspace/Modules/dummy/sed/run_preproc.sh', audio_path])
        
        with open('/workspace/Modules/dummy/sed/input/files.txt','r') as infile:
            files = infile.readlines()
            for i in range(len(files)):
                file = files[i].replace('\n','')
                feat_log = sed.feature(file)
                idx = int(file.split('/')[-1].replace('.wav',''))
                thres = 0.0
                out_dict = sed.process(idx, thres, feat_log, model)
                result_list.append(out_dict)
                print(out_dict)
        result_dict = {'audio_result': result_list}
        result = result_dict

        cmd = 'rm /workspace/Modules/dummy/sed/input/files.txt /workspace/Modules/dummy/sed/input/*.wav /workspace/Modules/dummy/sed/files/*.wav'
        subprocess.call(cmd, shell=True)
                
        return result

    def inference_by_text(self, data, video_info):
        result = []
        # TODO
        #   - Inference using image path
        #   -
        result = {"text_result": [
            {
                # 1 timestamp & multiple class
                'label': [
                    {'score': 1.0, 'description': 'word_name'},
                    {'score': 1.0, 'description': 'word_name'}
                ],
            },
            {
                # 1 timestamp & 1 class
                'label': [
                    {'score': 1.0, 'description': 'word_name'}
                ],
            }
        ]}
        self.result = result

        return self.result

if __name__ == "__main__":
    audio_path = '/workspace/Modules/dummy/sed/input/input.mp3'
    Dummy.inference_by_audio(audio_path)
