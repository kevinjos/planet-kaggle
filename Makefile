GRAPHDIR=graph
MODELDIR=model
INPUTDIR=input
OUTPUTDIR=output
LOGDIR=log

clean:
	rm -rf $(GRAPHDIR)/*
	rm -rf $(MODELDIR)/*

init:
	mkdir $(GRAPHDIR)
	mkdir $(MODELDIR)
	mkdir $(INPUTDIR)
	mkdir $(OUTPUTDIR)
	mkdir $(LOGDIR)
