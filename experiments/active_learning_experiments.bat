REM SENTIMENT Airline,GOP,Healthcare,Obama,SemEval
SET TASK_KEY=SENTIMENT
SET DATA_KEY=Airline
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

SET DATA_KEY=GOP
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

SET DATA_KEY=Healthcare
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

SET DATA_KEY=Obama
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional


SET DATA_KEY=SemEval
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

SET DATA_KEY=Clarin
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top --max-iters 200
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional --max-iters 200

REM ABUSIVE Founta,WaseemSRW
SET TASK_KEY=ABUSIVE
SET DATA_KEY=Founta
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

SET DATA_KEY=WaseemSRW
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

REM UNCERTAINITY Riloff,Swamy
SET TASK_KEY=UNCERTAINITY
SET DATA_KEY=Riloff
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional

SET DATA_KEY=Swamy
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection random
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring entropy --selection proportional
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection top
python SocialMediaIE\scripts\active_learning_experiment.py --task-key %TASK_KEY% --data-key %DATA_KEY% --scoring min_margin --selection proportional
