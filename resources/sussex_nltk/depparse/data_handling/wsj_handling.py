'''
Created on Jul 20, 2012

@author: adr27
'''
import os

def agglomerate_sections(section_list,path_to_wsj,outfile):
    section_set = set(section_list)
    def sort_key(name):
        try:
            return int(name[4:-4])
        except ValueError:
            return 0
    with open(outfile,'w') as out:
        for filename in sorted(os.listdir(path_to_wsj),key=sort_key):
            if filename.startswith('wsj'):
                section = int(filename[4:-6])
                if section in section_set:
                    print filename
                    with open(os.path.join(path_to_wsj,filename)) as infile: 
                        out.write(infile.read())
                        
def agglomerate_training(path_to_wsj,outfile):
    agglomerate_sections([i for i in xrange(2,22)], path_to_wsj, outfile)
    
def agglomerate_development(path_to_wsj,outfile):
    agglomerate_sections([22], path_to_wsj, outfile)
    
def agglomerate_testing(path_to_wsj,outfile):
    agglomerate_sections([0,1,23,24], path_to_wsj, outfile)
        

if __name__ == '__main__':
    pass
    #outfile = '/Volumes/LocalScratchHD/test/annotated_deps/annotation_repo/testagglom.txt'
    #agglomerate_sections([2,3],'/Volumes/research/calps/data3/scratch/adr27/DEPPARSE/treebank3-npbrac-stanforddeps-conll-twittertags',outfile)
    #agglomerate_training('/Volumes/research/calps/data3/scratch/adr27/DEPPARSE/treebank3-npbrac-stanforddeps-conll','/Volumes/research/calps/data3/scratch/adr27/DEPPARSE/agglomerated_gs/treebank3-npbrac-stanforddeps-conll-training.txt')
    agglomerate_development('/Volumes/research/calps/data3/scratch/adr27/DEPPARSE/treebank3-npbrac-stanforddeps-conll','/Volumes/research/calps/data3/scratch/adr27/DEPPARSE/agglomerated_gs/treebank3-npbrac-stanforddeps-conll-development.txt')