import ast
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.classification.policy import compute_metrics_dict, evaluate_probabilities, predictions, sweep_threshold, POLICY
from src.classification.preprocessing import bounds, crop_rgb, contract
from src.classification.dataset import load_sample_image, get_transforms
from src.classification.provenance import digest, file_hash, write_json, ROOT
from src.classification.models import build_classifier
from src.classification.train import effective_trial
from src.classification.reporting import save_records, summarize_records
from src.classification.smoke import assets
from src.data_prep import map_graph_w_name, exact_split, normalize_board_id, validate_manifest, membership
from inference_classification import build_standalone_classifier, get_inference_transform, load_model, validated_bounds, run_full_board_inference

torch.set_num_threads(2)
(ROOT/"tmp").mkdir(exist_ok=True)

class MetricsTests(unittest.TestCase):
    def test_fixed_25_absent_and_score(self):
        m = compute_metrics_dict([0,0,1], [0,1,1])
        self.assertEqual(len(m['support']),25)
        self.assertEqual(m['support'][:3],[2,1,0])
        self.assertEqual(m['zero_support_classes'],list(range(2,25)))
        for macro, per in [('P_macro','P_per_class'),('R_macro','R_per_class'),('F1_macro','F1_per_class')]:
            self.assertAlmostEqual(m[macro],sum(m[per])/25)
        self.assertAlmostEqual(m['S_cls'],.2*m['P_macro']+.2*m['R_macro']+.6*m['F1_macro'])

    def test_threshold_rejection_and_rows(self):
        p = np.full((2,25),.6/24); p[0,0]=.4; p[1,1]=.4
        a,t=predictions(p,0)
        np.testing.assert_array_equal(a,t)
        m = evaluate_probabilities(p,[0,1],.9)
        self.assertEqual(m['argmax']['coverage'],1)
        self.assertEqual(m['frozen_threshold']['coverage'],0)
        self.assertEqual(m['frozen_threshold']['rejected_count'],2)
        self.assertEqual(m['frozen_threshold']['R_macro'],0)
        self.assertEqual(m['frozen_threshold']['F1_macro'],0)
        self.assertEqual(np.array(m['argmax']['confusion_matrix']).shape,(25,25))
        cm = np.array(m['frozen_threshold']['confusion_matrix'])
        self.assertEqual(cm.shape,(25,26))
        np.testing.assert_array_equal(cm.sum(1),m['frozen_threshold']['support'])
        self.assertEqual(cm[0,-1],1)
        self.assertEqual(sweep_threshold(p,[0,1])[-1],0.)
        half = compute_metrics_dict([0,0],[0,-1])
        self.assertAlmostEqual(half['R_per_class'][0],.5)
        self.assertAlmostEqual(half['F1_per_class'][0],2/3)

    def test_bad_probability_and_label_inputs(self):
        for p in [np.zeros((2,25)), np.ones((1,24)), np.full((1,25),np.nan)]:
            with self.assertRaises(ValueError): predictions(p)
        with self.assertRaises(ValueError): compute_metrics_dict([25],[0])

class MappingTests(unittest.TestCase):
    def test_regressions(self):
        cases={'QA':24,'QA1':24,'JP':25,'JP1':25,'jumper':25,'jumper JP1':25,
               'clock':None,'clock X1':None,'potentiometer':1,'potentiometer P1':1,
               'display':None,'display D1':None,'U1':3,'IC1':11,'ic U1':11,
               'connector TP1':5,'resistor network RN1':9,'RA1':7,'CRA1':17,
               'RN1':9,'LED1':22,'S1':23,'SW1':18,'random':None,'cable':None,
               '"electrolytic capacitor" C1':2,'zener diode Z1':8,
               'Integrated Circuit (U)':3,'Integrated Circuit (IC)':11}
        for raw,want in cases.items():
            with self.subTest(raw=raw): self.assertEqual(map_graph_w_name(raw),want)

    def test_exact_split_and_overlap(self):
        self.assertEqual(exact_split('board10',['board1'],['board10']),'test')
        self.assertEqual(normalize_board_id('graph_12_BOARD1.pt'),'board1')
        with self.assertRaises(ValueError): exact_split('board100',['board1'],['board10'])
        with self.assertRaises(ValueError): exact_split('board1',['board1'],['BOARD1'])

    def test_official_completeness_and_frozen_validation(self):
        from src.data_prep import verify_official, derive
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            mp,cfg=assets(Path(temp)/'assets')
            m=json.loads(mp.read_text())
            root=Path(temp)/'official'
            for ds in ['Graph-W','Graph-F']:
                for split in ['train','test']:
                    (root/ds/'graphs'/split).mkdir(parents=True)
            for bid in ['train0','train1','val0','val1']:
                (root/'Graph-W'/'graphs'/'train'/('graph_1_'+bid+'.pt')).touch()
            status=verify_official(m,root)
            self.assertTrue(all(s.startswith('verified') for s in status.values()))
            (root/'Graph-W'/'graphs'/'train'/'extra.pt').touch()
            with self.assertRaises(ValueError): verify_official(m,root)

class DataConfigTests(unittest.TestCase):
    def test_crop_cache_and_stale_detection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as temp:
            mp,cfg=assets(Path(temp)/'assets')
            m=validate_manifest(mp)
            s=m['train_crops'][0]
            cached=load_sample_image(s)
            source=load_sample_image(dict(s,storage='source'))
            np.testing.assert_array_equal(cached,source)
            np.testing.assert_array_equal(get_transforms(32,False)(cached),get_inference_transform(32)(source))
            self.assertEqual(bounds(s['bbox'],(40,40)),validated_bounds(s['bbox'],(40,40)))
            Image.new('RGB',(2,2)).save(s['crop_path'])
            with self.assertRaises(ValueError): load_sample_image(s)
            with self.assertRaises(FileNotFoundError): load_sample_image(dict(s,crop_path=str(Path(temp)/'missing.png')))
        for b in [[-1,0,4,4],[0,0,0,1],[0,0,41,2],[0,0,float('nan'),2]]:
            with self.assertRaises(ValueError): bounds(b,(40,40))

    def test_config_propagation(self):
        c=effective_trial(ROOT/'config.yml','trial_3_convnext_tiny',epochs=1,pretrained=False)
        self.assertEqual(c['weight_decay'],.05)
        self.assertEqual(c['learning_rate'],.0001)
        self.assertEqual(c['epochs'],1)
        self.assertEqual(c['img_size'],224)
        self.assertEqual(c['batch_size'],32)
        self.assertEqual(c['imbalance'],'combined')
        self.assertFalse(c['pretrained'])

    def test_exclusive_outputs(self):
        from src.classification.provenance import new_run
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            out,rid=new_run(Path(temp)/'run')
            with self.assertRaises(FileExistsError): new_run(out)
            write_json(out/'metrics.json',{'a':1})
            with self.assertRaises(FileExistsError): write_json(out/'metrics.json',{'a':2})

    def test_manifest_consistency_and_cache_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            mp,cfg=assets(Path(temp)/'assets')
            m=json.loads(mp.read_text())
            m['train_crops'][0]['cache_id']='stale'
            bad=Path(temp)/'bad.json'; write_json(bad,m)
            with self.assertRaises(ValueError): validate_manifest(bad,False)

class CheckpointTests(unittest.TestCase):
    def test_three_architectures_and_standalone_round_trip(self):
        for name in ['efficientnet_b0','mobilenet_v3_large','convnext_tiny']:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=ROOT/"tmp") as temp:
                model=build_classifier(name,pretrained=False).eval()
                x=torch.randn(2,3,32,32)
                with torch.no_grad(): expected=model(x)
                mapping=json.loads((ROOT/'data_mapping.json').read_text())
                meta={'schema':'classification-checkpoint-v2','class_names':[c['name'] for c in mapping['classes']],
                      'preprocessing':contract(32),'effective_config':{'dropout_rate':.3}}
                ck=Path(temp)/'model.ckpt'
                torch.save({'model_name':name,'model_state_dict':model.state_dict(),'metadata':meta,
                            'threshold':.71,'decision_policy':POLICY},ck)
                standalone,got_name,threshold=load_model(str(ck),torch.device('cpu'))
                self.assertEqual(threshold,.71); self.assertEqual(got_name,name)
                with torch.no_grad(): actual=standalone(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                with self.assertRaises(ValueError):
                    image=Path(temp)/'image.png'; Image.new('RGB',(40,40)).save(image)
                    run_full_board_inference(str(image),str(ck),str(Path(temp)/'out.png'))
                rows=run_full_board_inference(str(image),str(ck),str(Path(temp)/'board.png'),boxes=[[0,0,20,20],[10,10,40,40]])
                self.assertEqual(len(rows),2)
                self.assertTrue(all(r['accepted']==(r['confidence']>=.71) for r in rows))

    def test_notebook_script_numerical_agreement(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as temp:
            p=np.full((2,25),.5/24); p[0,0]=.5;p[1,1]=.5
            samples=[dict(sample_id=str(i),board_id='synthetic',dataset='synthetic',image_path='synthetic.png',bbox=[0,0,1,1],class_id=i+1) for i in range(2)]
            path=Path(temp)/'predictions.json'
            save_records(path,samples,p,[0,1],.71,{},'val')
            nb=json.loads((ROOT/'task1results.ipynb').read_text())
            source=''.join(nb['cells'][1]['source'])
            tree=ast.parse(source)
            func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize_prediction_file')
            namespace={'summarize_records':summarize_records}
            exec(compile(ast.Module(body=[func],type_ignores=[]),'notebook','exec'),namespace)
            self.assertEqual(namespace['summarize_prediction_file'](path),summarize_records(path))

if __name__=='__main__': unittest.main()
