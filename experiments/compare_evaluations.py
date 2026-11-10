"""Compare user-supplied saved evaluations offline, pairing whole questions."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.behavioral_metrics import load


def compare(before, after, draws=10000):
    """Return per-question metric differences and paired bootstrap intervals.

    Each argument is a list of completed run directories; disjoint saved slices
    can be joined without regenerating paid attempts. Never silently drop failures
    or unmatched question/repetition IDs. Five attempts per question are required.
    """
    populations=[]
    reference=None
    for directories in (before, after):
        records={}
        seeds={}
        for directory in directories:
            import csv
            with (directory/'validation/results.csv').open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    key=(row['source_question_id'],int(row['repetition_index']))
                    if key in seeds: raise ValueError('Overlapping evaluation slices')
                    seeds[key]=(row['sampling_seed'],row['question'],row['correct_answer'],row['gold_document_ids'])
            for row in load(directory):
                key=(row['question_id'],row['repetition'])
                if key in records: raise ValueError('Duplicate attempt')
                records[key]=row['metrics']
        if reference is None: reference=seeds
        elif seeds!=reference: raise ValueError('Question/reference/seed mismatch')
        questions=sorted({q for q,r in records})
        if len(records)!=len(questions)*5 or any((q,r) not in records for q in questions for r in range(5)):
            raise ValueError('Require five complete attempts per question')
        names=list(next(iter(records.values())))
        cube=np.array([[[records[q,r][k] for k in names] for r in range(5)] for q in questions])
        populations.append(np.column_stack((cube.mean(axis=1),cube[:,:,names.index('em')].max(axis=1))))
    names=names+['pass_at_5']
    a,b=populations
    delta=b-a
    rng=np.random.default_rng(42)
    boot=[]
    rate_specs={'no_new_search_rate':('no_new_searches','searches'),
                'protocol_error_turn_rate':('protocol_errors','turns'),
                'after_coverage_search_rate':('searches_after_coverage','searches')}
    rate_boot={key:[] for key in rate_specs}
    for start in range(0,draws,100):
        indices=rng.integers(0,len(delta),size=(min(100,draws-start),len(delta)))
        boot.append(delta[indices].mean(axis=1))
        for key,(numer,denom) in rate_specs.items():
            ni,di=names.index(numer),names.index(denom)
            x=a[:,[ni,di]][indices].sum(axis=1)
            y=b[:,[ni,di]][indices].sum(axis=1)
            rate_boot[key].append(y[:,0]/y[:,1]-x[:,0]/x[:,1])
    bounds=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    return {'questions':len(a),'attempts_per_question':5,
            'method':f'{draws:,} paired whole-question percentile bootstrap resamples; seed 42; no training-seed uncertainty or multiplicity adjustment',
            'pooled_rates':{key:{'before':float(a[:,names.index(n)].sum()/a[:,names.index(d)].sum()),
                                'after':float(b[:,names.index(n)].sum()/b[:,names.index(d)].sum()),
                                'difference_ci95':np.quantile(np.concatenate(rate_boot[key]),[.025,.975]).tolist()}
                            for key,(n,d) in rate_specs.items()},
            'metrics':{k:{'before':float(a[:,j].mean()),'after':float(b[:,j].mean()),
                          'difference':float(delta[:,j].mean()),'ci95':bounds[:,j].tolist()} for j,k in enumerate(names)}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,nargs='+',required=True)
    parser.add_argument('--after',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=Path('data/analysis/comparison.json'))
    args=parser.parse_args()
    result=compare(args.before,args.after)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
