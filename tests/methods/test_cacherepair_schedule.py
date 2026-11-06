import json
import random

import pytest
import torch

from methods.CacheRepair.qwen38.scheduled_validation import ScheduledValidation,evaluation_state,request_epoch_quality


def test_schedule_is_durable_complete_and_does_not_reset_on_restart(tmp_path):
    now=[100.]
    rows=[{'chunks':[[1],[2]]}]*500
    plan=dict(enabled=True,hourly_validation_examples=500,hourly_interval_seconds=3600)
    schedule=ScheduledValidation(tmp_path,rows,plan,clock=lambda:now[0])
    assert not schedule.due() and not schedule.state['baseline_done']
    with pytest.raises(ValueError,match='Incomplete'):
        schedule.record(0,'baseline',dict(examples=499),1.)
    schedule.record(0,'baseline',dict(examples=500),1.)
    now[0]=3699.
    resumed=ScheduledValidation(tmp_path,rows,plan,clock=lambda:now[0])
    assert resumed.state['baseline_done'] and not resumed.due()
    now[0]=3700.
    assert resumed.due()
    resumed.record(12,'hourly',dict(examples=500),8.)
    assert resumed.state['next_due']==7300.
    assert json.loads((tmp_path/'scheduled-validation/hourly-step-00000012.json').read_text())['validation']['examples']==500
    with pytest.raises(ValueError,match='changed'):
        ScheduledValidation(tmp_path,list(reversed(rows[:-1]))+[{'chunks':[[3],[4]]}],plan)


def test_evaluation_preserves_rng_modes_and_parameters_even_on_failure():
    model=torch.nn.Linear(2,2).train()
    before={k:v.clone() for k,v in model.state_dict().items()}
    cpu=torch.get_rng_state();py=random.getstate()
    with pytest.raises(RuntimeError):
        with evaluation_state(model):
            assert not model.training and not torch.is_grad_enabled()
            random.random();torch.rand(12)
            raise RuntimeError('test')
    assert model.training and torch.equal(torch.get_rng_state(),cpu) and random.getstate()==py
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_epoch_request_survives_crash_and_only_verified_completion_skips(tmp_path):
    output=tmp_path/'training';output.mkdir()
    checkpoint=output/'checkpoint-12'
    plan=dict(enabled=True,epoch_quality=dict(enabled=True))
    training=dict(checkpoint_epoch=1,checkpoint_step=12)
    assert request_epoch_quality(output,checkpoint,training,plan)
    assert request_epoch_quality(output,checkpoint,training,plan)
    done=tmp_path/'quality-evaluations/step-00000012/completed.json'
    done.parent.mkdir(parents=True)
    done.write_text(json.dumps(dict(status='completed',checkpoint=str(checkpoint.resolve()))))
    assert not request_epoch_quality(output,checkpoint,training,plan)
    done.write_text(json.dumps(dict(status='completed',checkpoint='wrong')))
    with pytest.raises(ValueError,match='differs'):
        request_epoch_quality(output,checkpoint,training,plan)


def test_missing_quality_completion_retries_in_supervisor(tmp_path):
    from scripts.RunCacheRepairTrainingFast import quality_request
    output=tmp_path/'training';output.mkdir()
    assert quality_request(tmp_path) is None
    (output/'epoch-quality-request.json').write_text('{"step": 12, "checkpoint": "checkpoint"}')
    assert quality_request(tmp_path)==output/'epoch-quality-request.json'
    done=tmp_path/'quality-evaluations/step-00000012/completed.json'
    done.parent.mkdir(parents=True);done.write_text('{"status":"completed","checkpoint":"checkpoint"}')
    assert quality_request(tmp_path) is None


@pytest.mark.parametrize('evaluation_fails',[False,True])
def test_supervisor_runs_epoch_quality_then_exact_resume_or_stops_on_failure(tmp_path,monkeypatch,evaluation_fails):
    from scripts import RunCacheRepairTrainingFast as runner
    from scripts import EvaluateCacheRepairEpoch as quality
    job=tmp_path/'job';data=tmp_path/'data';job.mkdir();data.mkdir()
    (data/'dataset.json').write_text('{}')
    config=tmp_path/'config.json';config.write_text('{}')
    plan=tmp_path/'eval.json';plan.write_text('{"enabled":true}')
    (job/'fast-validation-native-norm.json').write_text('{"status":"passed"}')
    (job/'six-gpu-smoke.json').write_text('{"passed":true}')
    calls=[]
    class Child:
        returncode=0
        def __init__(self,command,**kwargs):
            calls.append(command)
            training=job/'training';training.mkdir(exist_ok=True)
            if len(calls)==1:
                (training/'status.json').write_text('{"status":"interrupted"}')
                (training/'interrupted.json').write_text('{"reason":"epoch_quality_evaluation"}')
                (training/'latest.json').write_text('{"step":3}')
                (training/'epoch-quality-request.json').write_text('{"step":3,"checkpoint":"checkpoint"}')
            elif len(calls)==2:
                if evaluation_fails:self.returncode=1
                else:
                    done=job/'quality-evaluations/step-00000003/completed.json'
                    done.parent.mkdir(parents=True)
                    done.write_text('{"status":"completed","checkpoint":"checkpoint"}')
            else:
                (training/'status.json').write_text('{"status":"completed"}')
        def poll(self):return self.returncode
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.sys,'argv',['runner','--job',str(job),'--model','target','--dataset',str(data),
        '--config',str(config),'--physical-gpus','0,1','--evaluation-plan',str(plan)])
    monkeypatch.setattr(quality,'verify_quality_inputs',lambda plan:None)
    monkeypatch.setattr(runner,'gpu_snapshot',lambda:[])
    monkeypatch.setattr(runner,'available_gpus',lambda *args:[0,1])
    monkeypatch.setattr(runner.fcntl,'flock',lambda *args:None)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    if evaluation_fails:
        with pytest.raises(RuntimeError,match='quality failed'):runner.main()
        assert len(calls)==2
        assert json.loads((job/'training/latest.json').read_text())['step']==3
    else:
        runner.main()
        assert len(calls)==3 and '--resume' in calls[2] and calls[2][-1]=='auto'
        assert json.loads((job/'job-status.json').read_text())['status']=='completed'
    assert '--evaluation-plan' in calls[0] and 'EvaluateCacheRepairEpoch.py' in calls[1][1]
