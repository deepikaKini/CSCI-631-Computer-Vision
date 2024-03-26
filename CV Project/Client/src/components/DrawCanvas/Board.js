import React from "react";
import "./DrawCanvas.css";
import p5 from "p5";
import * as ml5 from "ml5";
import { useEffect, useRef, useState } from "react";
import * as rdp from "./rdp";

// Board component
function Board(parentState) {
  // State for setup status
  const [isSetupDone, setIsSetupDone] = useState(false);

  // Extracting state and functions from the parentState
  const clearBoard = parentState.clearBoard;
  const setClearBoard = parentState.setClearBoard;
  const doodleModel = parentState.doodleModel;
  const setStatusText = parentState.setStatusText;
  
  let imageIndex = 0;

  // Variables related to sketchRNN and drawing
  let sketchRNN;
  let currentStroke;
  let x, y;
  let nextPen = "down";
  let seedPath = [];

  // Variables to store points and drawing status
  let points = [];
  let personDrawing = false;
  let sketchRNNDrawStart = false;
  var check =0
  const saveCanvasImage = (p5) => {
    p5.saveCanvas(`image_${imageIndex}`, "png");
    imageIndex += 1;
  };
  

  // Function to start sketchRNN drawing
  const sketchRNNStart = (p5) => {
    personDrawing = false;
    sketchRNNDrawStart = true;
  };

  // Function to preload the sketchRNN model
  const preload = (doodleModel) => {
    sketchRNN = ml5.sketchRNN(doodleModel);
    console.log("model loaded");
    setStatusText("Model Loaded!");
    console.log(doodleModel);
    console.log("Api call here");
  };

  // Function to start user drawing
  const startDrawing = () => {
    personDrawing = true;
    x = p5.mouseX;
    y = p5.mouseY;
  };

  // Callback function when sketchRNN generates a stroke path
  const gotStrokePath = (error, strokePath) => {
    currentStroke = strokePath;
  };

  // Fetch points data from the server when the component mounts
  useEffect(() => {
    fetch("http://127.0.0.1:5000/here")
      .then((response) => response.json())
      .then((json) => {
        // Convert the JSON data
        const transformedData = json.points.map((item) => {
          const [point1, point2] = item;
          const [x1, y1] = point1;
          return {
            x: x1,
            y: y1,
          };
        });

        points = transformedData;
        console.log(transformedData);
      })
      .catch((error) => console.error(error));
  }, []);

  // Sketch function that defines the p5 drawing
  const Sketch = (p5) => {
    console.log("Here")
    let radius;
    console.log("check")
    // Check if setup is done, and if not, set up the canvas
    if (!isSetupDone) {
      setIsSetupDone(true);
      p5.setup = () => {
        let canvas = p5.createCanvas(1200, 1200);
        p5.background(0);
        canvas.mousePressed(startDrawing);
        canvas.mouseReleased(sketchRNNStart);
        radius = 0;
      };
    }

    // Draw function that is called in a loop
    console.log("Here2")
    p5.draw = () => {
      // console.log("Here3")
      p5.stroke(255);
      p5.strokeWeight(4);

      // Clear the board if needed
      if (clearBoard) {
        console.log("Inside clear Board");
        setClearBoard(false);
        currentStroke = false;
        x = undefined;
        y = undefined;
        nextPen = "down";
        seedPath = [];

        points = [];
        personDrawing = false;
        sketchRNNDrawStart = false;
        p5.background(0);
        console.log("Here3")
      }

      // Start sketchRNN drawing if the flag is set
      if (sketchRNNDrawStart && check < 5) {
        
        const convertedPoints = points.map((point) =>
          p5.createVector(point.x - 50, point.y - 50)
        );

        // Draw the original path
        for (let i = 0; i < points.length - 1; i++) {
          p5.line(
            points[i].x - 100,
            points[i].y,
            points[i + 1].x,
            points[i + 1].y
          );
        }

        // Simplify the path using the Ramer-Douglas-Peucker algorithm
        const rdpPoints = [];
        const total = convertedPoints.length;
        const start = convertedPoints[0];
        const end = convertedPoints[total - 1];
        rdpPoints.push(start);

        rdp.rdp(0, total - 1, convertedPoints, rdpPoints, p5);
        rdpPoints.push(end);

        // Drawing simplified path
        p5.background(0);
        p5.stroke(255);
        p5.strokeWeight(4);
        p5.beginShape();
        p5.noFill();
        for (let v of rdpPoints) {
          p5.vertex(v.x, v.y);
        }
        p5.endShape();

        // Update current position for sketchRNN
        x = rdpPoints[rdpPoints.length - 1].x;
        y = rdpPoints[rdpPoints.length - 1].y;

        // Clear seedPath
        seedPath = [];

        // Converting to SketchRNN states
        for (let i = 1; i < rdpPoints.length; i++) {
          let strokePath = {
            dx: rdpPoints[i].x - rdpPoints[i - 1].x,
            dy: rdpPoints[i].y - rdpPoints[i - 1].y,
            pen: "down",
          };
          seedPath.push(strokePath);
        }

        // Generate stroke path with sketchRNN
        sketchRNN.generate(seedPath, gotStrokePath);
        sketchRNNDrawStart = false;
        
        check = check + 1

      }

      // Continue drawing if there is a current stroke
      if (currentStroke) {
        
        if (nextPen == "end") {
          console.log("Here4")
          
          sketchRNN.reset();
          sketchRNNStart();
          currentStroke = null;
          nextPen = "down";
          saveCanvasImage(p5)
          return;
        }

        if (nextPen == "down") {
         
          p5.line(x, y, x + currentStroke.dx, y + currentStroke.dy);
        }
        x += currentStroke.dx;
        y += currentStroke.dy;
        nextPen = currentStroke.pen;
        currentStroke = null;

        // Generate next stroke path with sketchRNN
        sketchRNN.generate(gotStrokePath);
       
      }
      

    };

  };

  // UseEffect to preload the sketchRNN model and initialize the p5 sketch
  useEffect(() => {
    preload(doodleModel);
    new p5(Sketch, "drawArea");
  }, []);

  // Render the component with a div for the p5 sketch
  return <div id="drawArea" className="draw-area"></div>;
}

export default Board;
