import subprocess
import sys
from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print("Qt: v", QT_VERSION_STR, "\tPyQt: v", PYQT_VERSION_STR)



def menue(sysargv1):
#  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< Mask an image >>3<< Mask Invert >>4<< Add2images(fit)  \n>>5<< Split tricolor >>6<< Combine Tricolor >>7<< Create Luminance(2ax) >>8<< Align2img \n>>9<< Plot_16-bit_img to 3d graph(2ax) >>10<< Centroid_Custom_filter(2ax) >>11<< UnsharpMask \n>>12<< FFT-(RGB) >>13<< Img-DeconvClr >>14<< Centroid_Custom_Array_loop(2ax) \n>>15<< Erosion(2ax) >>16<< Dilation(2ax) >>17<< DynamicRescale(2ax) >>18<< GausBlur  \n>>19<< DrCntByFileType >>20<< ImgResize >>21<< JpgCompress >>22<< subtract2images(fit)  \n>>23<< multiply2images >>24<< divide2images >>25<< max2images >>26<< min2images \n>>27<< imgcrop >>28<< imghiststretch >>31<< Video \n>>32<< gammaCor >>33<< ImgQtr >>34<< CpyOldHdr >>35<< DynReStr(RGB) \n>>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance >>40<< EdgeDetect \n>>41<< Mosaic(4) >>42<< BinImg >>43<< autostr >>44<< LocAdapt >>45<< WcsOvrlay \n>>46<< Stacking >>47<< CombineLRGB >>48<< MxdlAstap >>49<< CentRatio >>50<< ResRngHp \n>>51<< CombBgrAlIm >>52<< PixelMath >>53<< Color >>54<< ImageFilters \n>>1313<< Exit --> ")
  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< MaskOrMskInvImg >>7<< FitsSplit >>9<< Plot_img to 3d (2ax) \n>>10<< Centroid_Custom_filter(2ax) >>5<< DirSpltAllRgb >>8<< alingimg5kjpg \n>>14<< Centroid_Custom_Array_loop(2ax) >>17<< DynamicRescale(2ax) >>19<< DrCntByFileType \n>>>20<< ImgResize >>21<< JpgCompress >>27<< imgcrop >>28<< imghiststretch >>29<< gif \n>30<< aling2img(2pts) >>31<< Video >>32<< gammaCor >>33<< ImgQtr >>34<< CpyOldHdr \n>>35<< DynReStr(RGB) >>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance \n>>40<< EdgeDetect >>41<< Mosaic(4) >>42<< BinImg >>43<< autostr >>44<< Rank\n>>45<< WcsOvrlay >>46<< AlnImgsByDir >>47<< CombineLRGB >>48<< MxdlAstap >>49<< CentRatio \n>>51<< CombBgrAlIm >>52<< PixelMath >>53<< Color >>54<< ImageFilters >>55<< AlignImgs \n>>56<< Stacker >>57<< FitQc >>58<< Normalize >>59<< RaDec2ptAng\n>>1313<< Exit --> ")

  return sysargv1

sysargv1 = ''
while not sysargv1 == '1313':  # Substitute for a while-True-break loop.
  sysargv1 = ''
  sysargv1 = menue(sysargv1)

  if sysargv1 == '1':
      # read+exec with utf-8
      with open("fncts/affine_transform.py", "r", encoding="utf-8") as f:
          code = f.read()
      exec(code, globals(), locals())
      menue(sysargv1)

  if sysargv1 == '2':
      subprocess.Popen([sys.executable, "fncts/mask_tool_gui.py"])

  if sysargv1 == '5':
      subprocess.Popen([sys.executable, "fncts/fits_splitter.py"])

  if sysargv1 == '6':
    combinetricolor()

  if sysargv1 == '7':
      subprocess.Popen([sys.executable, "fncts/fits_splitter.py"])

  if sysargv1 == '8':
      subprocess.Popen([sys.executable, "fncts/align_jpg_gui.py"])

  if sysargv1 == '9':
      subprocess.Popen([sys.executable, "fncts/plot3d_gui.py"])

  if sysargv1 == '10':
    sysargv2  = input("Enter the file name -->")
    sysargv3 = input("Enter the radius of the file-->")
    sysargv4 = input("Enter the x-coordinate of the centroid-->")
    sysargv5 = input("Enter the y-coordinate of the centroid-->")
    PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5)
    menue(sysargv1)

  if sysargv1 == '11':
    unsharpMask()

  if sysargv1 == '12':
    FFT()

  if sysargv1 == '13':
    LrDeconv()

  if sysargv1 == '14':
    sysargv2  = input("Enter the file name -->")
    sysargv3 = input("Enter the radius of the file-->")
    sysargv4 = input("Enter the x-coordinate of the centroid-->")
    sysargv5 = input("Enter the y-coordinate of the centroid-->")
    x = int(sysargv4)
    y = int(sysargv5)
    for i in range(3):
      i1=int(i + x - 1)
      file = str(i1)
      for j in range(3):
        j1=int(j + y - 1)
        file1 = str(j1)
        sysargv4 = file
        sysargv5 = file1
        PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5)
    menue(sysargv1)
 
  if sysargv1 == '15':
    erosion()
 
  if sysargv1 == '16':
    dilation()

  if sysargv1 == '17':
      subprocess.Popen([sys.executable, "fncts/dynamic_rescale16_gui_cython.py"])

  if sysargv1 == '18':
    gaussian()

  if sysargv1 == '19':
      subprocess.Popen([sys.executable, "fncts/filecount_gui.py"])

  if sysargv1 == '20':
      subprocess.Popen([sys.executable, "fncts/resize_gui.py"])

  if sysargv1 == '21':
      subprocess.Popen([sys.executable, "fncts/jpgcomp_gui.py"])

  if sysargv1 == '22':
    subtract2images()

  if sysargv1 == '23':
    multiply2images()

  if sysargv1 == '24':
    divide2images()

  if sysargv1 == '25':
    max2images()

  if sysargv1 == '26':
    min2images()

  if sysargv1 == '27':
    imgcrop1()

  if sysargv1 == '28':
      subprocess.Popen([sys.executable, "fncts/imghiststretch_gui.py"])

  if sysargv1 == '31':
      subprocess.Popen([sys.executable, "fncts/video_gui.py"])

  if sysargv1 == '32':
      subprocess.Popen([sys.executable, "fncts/gamma_gui.py"])

  if sysargv1 == '33':
    imgqtr()

  if sysargv1 == '34':
      subprocess.Popen([sys.executable, "fncts/cpy_old_hdr_gui.py"])   

  if sysargv1 == '35':
    DynamicRescale16RGB()

  if sysargv1 == '36':
      subprocess.Popen([sys.executable, "fncts/clahe_gui.py"])  

  if sysargv1 == '37':
      subprocess.Popen([sys.executable, "fncts/pm_hist_tool_gui.py"]) 

  if sysargv1 == '38':
      subprocess.Popen([sys.executable, "fncts/hist_match_gui.py"])
  
  if sysargv1 == '39':
      subprocess.Popen([sys.executable, "fncts/distance_gui.py"])
  
  if sysargv1 == '40':
      subprocess.Popen([sys.executable, "fncts/edgedetect_gui.py"])

  if sysargv1 == '41':
    mosaic()

  if sysargv1 == '42':
      subprocess.Popen([sys.executable, "fncts/binimg_gui.py"])

  if sysargv1 == '43':
      subprocess.Popen([sys.executable, "fncts/autostr_gui.py"])

  if sysargv1 == '44':
      # read+exec with utf-8
      with open("fncts/wrank.py", "r", encoding="utf-8") as f:
          code = f.read()
      exec(code, globals(), locals())
      menue(sysargv1)

  if sysargv1 == '45':
      subprocess.Popen([sys.executable, "fncts/fits_wcs_plotter_gui.py"])

  if sysargv1 == '46':
      subprocess.Popen([sys.executable, "fncts/align_imgs_gui_fallback.py"])

  if sysargv1 == '47':
      subprocess.Popen([sys.executable, "fncts/fits_lrgb_combine_gui.py"])

  if sysargv1 == '48':
      subprocess.Popen([sys.executable, "fncts/mxdl_astap_gui.py"])

  if sysargv1 == '49':
      subprocess.Popen([sys.executable, "cent_ratio_gui.py"])

  if sysargv1 == '51':
      subprocess.Popen([sys.executable, "fncts/combine_weighted_gui.py"])

  if sysargv1 == '52':
      # read+exec with utf-8
      subprocess.Popen([sys.executable, "fncts/pixelmath.py"])

  if sysargv1 == '53':
      subprocess.Popen([sys.executable, "fncts/color_tool.py"])

  if sysargv1 == '54':
      subprocess.Popen([sys.executable, "fncts/image_filters.py"])

  if sysargv1 == '55':
      subprocess.Popen([sys.executable, "fncts/align_imgs.py"])

  if sysargv1 == '56':
      subprocess.Popen([sys.executable, "fncts/stacker_gui.py"])

  if sysargv1 == '57':
      # read+exec with utf-8
      with open("fncts/analyze_fits_roundness_trails.py", "r", encoding="utf-8") as f:
          code = f.read()
      exec(code, globals(), locals())
      menue(sysargv1)

  if sysargv1 == '58':
      subprocess.Popen([sys.executable, "fncts/normalize_gui.py"])

  if sysargv1 == '59':
      subprocess.Popen([sys.executable, "fncts/radectwoptang_gui.py"])

  if sysargv1 == '1313':
    sys.exit()



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      