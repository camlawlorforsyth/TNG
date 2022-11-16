
import numpy as np

from PIL import Image

def combine() :
    
    paths = ['MWA_gradient_subID_14_snapNum_10.png',
             'MWA_gradient_subID_14_snapNum_20.png',
             'MWA_gradient_subID_14_snapNum_30.png',
             'MWA_gradient_subID_14_snapNum_40.png',
             'MWA_gradient_subID_14_snapNum_50.png',
             'MWA_gradient_subID_14_snapNum_60.png',
             'MWA_gradient_subID_14_snapNum_70.png',
             'MWA_gradient_subID_14_snapNum_80.png',
             'MWA_gradient_subID_14_snapNum_90.png']
    
    # open the data from those images
    images = [Image.open(image) for image in paths]
    
    # get the widths and heights to create an empty final image
    widths, heights = zip(*(i.size for i in images))
    final = Image.new('RGB', (np.sum(widths), np.max(heights)))
    
    # populate the final image with the input images
    x_offset = 0
    for im in images :
        final.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    
    # save the final image
    final.save('MWA_gradient_subID_14_physically-interesting-range.png')
    
    return

def concatenate_diagnostics(subIDs, masses) :
    
    # adapted from
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    
    # define the input directory
    inDir = 'output/'
    
    # loop through all the subIDs in the sample
    for subID, mass in zip(subIDs, masses) :
        
        # define the paths of the SFH, zeta, xi, and SFR density gradient images
        SFH = inDir + 'quenched_SFHs_without_lohi(t)/quenched_SFH_subID_{}.png'.format(subID)
        zeta = inDir + 'zeta(t)/zeta_subID_{}.png'.format(subID)
        xi = inDir + 'xi(t)/xi_subID_{}.png'.format(subID)
        SFR_density = inDir + 'SFR_density_gradients(t)/SFR_density_gradient_subID_{}.png'.format(subID)
        
        # get the paths for the subID of interest
        paths = [SFH, zeta, xi, SFR_density]
        
        # open the data from those images
        images = [Image.open(image) for image in paths]
        
        # get the widths and heights to create an empty final image
        widths, heights = zip(*(i.size for i in images))
        final = Image.new('RGB', (np.sum(widths), np.max(heights)))
        
        # populate the final image with the input images
        x_offset = 0
        for im in images :
            final.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        
        # save the final image
        outDir = 'output/diagnostics(t)/'
        outfile = outDir + 'diagnostics_subID_{}_logM_{}.png'.format(subID, mass)
        final.save(outfile)    
    
    return

test_IDs = [
    # satellites
    14, 41, 63878, 167398, 184946, 220605, 324126,
    # primaries
    545003, 547545, 548151, 556699, 564498,
    592021, 604066, 606223, 607654, 623367
    ]

test_masses = [
    # satellites
    10.441438, 10.100833, 10.331834, 10.367537, 10.231799, 10.138943, 10.334204,
    # primaries
    10.273971, 10.437312, 10.496466, 10.423783, 10.455939,
    10.303858, 10.334443, 10.264565, 10.438307, 10.261037
    ]

# concatenate_diagnostics(test_IDs, test_masses)
