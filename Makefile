GRAPHDIR=graph
MODELDIR=model
LOGDIR=log

clean:
	rm -rf $(GRAPHDIR)/atmos/*
	rm -rf $(GRAPHDIR)/clear/*
	rm -rf $(GRAPHDIR)/haze/*
	rm -rf $(GRAPHDIR)/partly-cloudy/*
	rm -rf $(MODELDIR)/*
	rm -rf $(LOGDIR)/*
