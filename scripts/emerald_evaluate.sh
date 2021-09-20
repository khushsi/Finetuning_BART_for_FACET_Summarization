TASK='hypo/'

python emerald_evaluate.py ${1}

export CLASSPATH=~/package/fairseq/stanford-corenlp-3.7.0.jar:~/package/fairseq/stanford-corenlp-4.2.0-models.jar:/mnt/efs/stanford-corenlp-4.2.0/*

# Tokenize hypothesis and target files.
entry=$TASK${1}".test.hypo"
cat $entry | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $entry".tokenized"
cat data/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $entry".target"
files2rouge  $entry".tokenized" $entry".target"

entry=$TASK${1}".dev.hypo"
cat $entry | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $entry".tokenized"
cat data/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $entry".target"
files2rouge  $entry".tokenized" $entry".target"

# Expeted output: (ROUGE-2 Average_F: 0.xxxx)