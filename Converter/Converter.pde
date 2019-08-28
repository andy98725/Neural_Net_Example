
final boolean convertImg = false;
final boolean dispOutFile = true;
void setup() {
  size(200, 200);
  if(convertImg){
  convertTraining("/training.png", "../input/training.txt");
  }
  if (dispOutFile) {
    product = convertImage("../NNets/biggerOutput/testOut.txt");
    surface.setResizable(true);
    surface.setSize(product.width, product.height);
  }
}

void draw() {
  if (dispOutFile){
    image(product, 0, 0);
  }
}

PImage product;
final boolean convertSize = false;
final boolean isInt = false;
final int conversionRate = 255;

void convertTraining(String filename, String savename) {
  PImage convert = loadImage(filename);
  String converted[];
  int index;
  if (convertSize) {
    converted = new String[convert.width*convert.height+1];
    converted[0] = str(convert.width) + ' ' + str(convert.height);
    index = 1;
  } else {
    converted = new String[convert.width*convert.height];
    index = 0;
  }
    println(convert.width);
    println(convert.height);
  convert.loadPixels();
  for (int j = 0; j < convert.height; j++) for (int i = 0; i < convert.width; i++) {
    if (!isInt)
      converted[index+j*convert.width + i] = str(i) + ' ' + str(j) + ' ' + str(red(convert.pixels[j*convert.width + i])/conversionRate) + ' ' + str(green(convert.pixels[j*convert.width + i])/conversionRate) + ' ' + str(blue(convert.pixels[j*convert.width + i])/conversionRate);
    else
      converted[index+j*convert.width + i] = str(i) + ' ' + str(j) + ' ' + str((int)red(convert.pixels[j*convert.width + i])/conversionRate) + ' ' + str((int)green(convert.pixels[j*convert.width + i])/conversionRate) + ' ' + str((int)blue(convert.pixels[j*convert.width + i])/conversionRate);
  }
  saveStrings(savename, converted);
}

PImage convertImage(String textfilename) {
  String[] out = loadStrings(textfilename);
  String[] scoords = out[0].split(" ", 2);
  PImage output = new PImage(int(scoords[0]), int(scoords[1]));
  output.loadPixels();
  for (int i = 1; i < out.length; i++) {
    String[] split = out[i].split(" ", 0);
    if (split.length > 4) {
      int x = int(split[0]), y = int(split[1]);
      float r = float(split[2]) * conversionRate, g = float(split[3]) * conversionRate, b = float(split[4]) * conversionRate;
      output.pixels[y * output.width + x] = color(r, g, b);
    } else {
      println("Error in " + textfilename + ": line number " + i);
      exit();
    }
  }
  output.updatePixels();
  return output;
}
