import os 
import shutil 
import argparse

def decompose(com,l_chain, out_path):
    basename = os.path.basename(com)
    dirname = os.path.join(out_path, (os.path.splitext(basename)[0] + '_adcp'))
    l_name = os.path.join(dirname, 'lig.pdb')
    r_name = os.path.join(dirname, 'rec.pdb')
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    com_ =os.path.join(dirname, basename)
    shutil.copy(com, com_)
    f = open(com_)
    lines = [i for i in f.readlines() if len(i) >= 21 and i[:4] == 'ATOM']
    f.close()
    f_l = open(l_name, 'w')
    f_r = open(r_name, 'w')
    for line in lines:
        if line[21] == l_chain:
            if ''.join([i for i in line[12:16].strip() if not i.isdigit()])[0] != 'H':
                f_l.write(line)
        else:
            if ''.join([i for i in line[12:16].strip() if not i.isdigit()])[0] != 'H':
                f_r.write(line)
    return l_name, r_name

def adcp_protocol(l:str, r:str, nproc):
    path0 = os.path.dirname(l)
    current_dir = os.getcwd()
    os.chdir(path0)
    lig = 'lig.pdb'
    p = 'ligPocket.trg'
    rec = 'rec.pdb'
    log = 'log.txt'
    os.system('reduce %s > %s &&'%(l, lig)+\
                'reduce %s > %s &&'%(r, rec)+\
                'prepare_ligand -l %s -o %s &&'%(lig, lig+'qt')+\
                'prepare_receptor -r %s -o %s &&'%(rec, rec+'qt')+ \
                'agfr -r %s -l %s -o %s -ng &&' %(rec+'qt', lig+'qt', p) + \
                'adcp -i %s -t %s -o adcp -N 10 -n 500000 -O -c %d > %s' %(lig, p, nproc, log))
    os.chdir(current_dir)

def arg_parser():
    parser = argparse.ArgumentParser(description='Norm the reccptor structure file')
    parser.add_argument('--receptor', '-r', dest='rec', default=False)
    parser.add_argument('--complex', '-c', dest='com', default=False)
    parser.add_argument('--ligand', '-l', dest='lig', default=False)
    parser.add_argument('--ligand_chain', '-lc', dest='lic', default='L')
    return parser
