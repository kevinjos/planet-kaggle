GRAPHDIR=graph
MODELDIR=model

clean:
	rm -rf $(GRAPHDIR)/atmos/*
	rm -rf $(GRAPHDIR)/clear/*
	rm -rf $(GRAPHDIR)/haze/*
	rm -rf $(GRAPHDIR)/partly-cloudy/*
	rm -rf $(MODELDIR)/atmos/*
	rm -rf $(MODELDIR)/clear/*
	rm -rf $(MODELDIR)/haze/*
	rm -rf $(MODELDIR)/partly-cloudy/*
