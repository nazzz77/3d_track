// Decode every frame of a Stray Scanner rgb.mp4 (HEVC) into resized JPEG files.
//
// macOS only. Uses the built-in AVFoundation decoder, so no extra install is needed.
//
// Build:  swiftc -O stray_decode.swift -o stray_decode
// Usage:  stray_decode <rgb.mp4> <out_dir> <width> <height> [jpeg_quality=0.9]
//
// Writes <out_dir>/000000.jpg, 000001.jpg, ... in presentation order,
// <out_dir>/pts.txt with one presentation timestamp (seconds) per frame, and
// <out_dir>/source_size.txt with the source video width and height.

import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count >= 5, let width = Int(args[3]), let height = Int(args[4]) else {
    FileHandle.standardError.write("usage: stray_decode <rgb.mp4> <out_dir> <width> <height> [jpeg_quality]\n".data(using: .utf8)!)
    exit(1)
}
let inURL = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2], isDirectory: true)
let quality = args.count > 5 ? (Double(args[5]) ?? 0.9) : 0.9
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let asset = AVURLAsset(url: inURL)
guard let track = try await asset.loadTracks(withMediaType: .video).first else {
    FileHandle.standardError.write("no video track in \(inURL.path)\n".data(using: .utf8)!)
    exit(1)
}
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
])
output.alwaysCopiesSampleData = false
reader.add(output)
guard reader.startReading() else {
    FileHandle.standardError.write("cannot start reading: \(String(describing: reader.error))\n".data(using: .utf8)!)
    exit(1)
}

let ciContext = CIContext(options: [.useSoftwareRenderer: false])
let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
var pts: [Double] = []
var srcSize = (0, 0)

func writeJPEG(_ image: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
        FileHandle.standardError.write("cannot create \(url.path)\n".data(using: .utf8)!)
        exit(1)
    }
    CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    guard CGImageDestinationFinalize(dest) else {
        FileHandle.standardError.write("failed to write \(url.path)\n".data(using: .utf8)!)
        exit(1)
    }
}

func tmpURL(_ i: Int) -> URL { outDir.appendingPathComponent(String(format: "decode_%06d.jpg", i)) }

// Frames are written as they are decoded (to keep memory low) and renamed into
// presentation order at the end.
while let sample = output.copyNextSampleBuffer() {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
    let t = CMSampleBufferGetPresentationTimeStamp(sample).seconds
    srcSize = (CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer))
    let ci = CIImage(cvPixelBuffer: pixelBuffer)
    let sx = CGFloat(width) / ci.extent.width
    let sy = CGFloat(height) / ci.extent.height
    // Lanczos downscaling; aspect ratio of 1920x1440 and 960x720 is identical (4:3).
    let scaled = ci.applyingFilter("CILanczosScaleTransform", parameters: [
        kCIInputScaleKey: sy,
        kCIInputAspectRatioKey: sx / sy,
    ])
    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    guard let cg = ciContext.createCGImage(scaled, from: rect, format: .RGBA8, colorSpace: colorSpace) else {
        FileHandle.standardError.write("failed to render frame at pts \(t)\n".data(using: .utf8)!)
        exit(1)
    }
    writeJPEG(cg, to: tmpURL(pts.count))
    pts.append(t)
}
if reader.status == .failed {
    FileHandle.standardError.write("decode failed: \(String(describing: reader.error))\n".data(using: .utf8)!)
    exit(1)
}

let order = pts.indices.sorted { pts[$0] < pts[$1] }
let reordered = order != Array(pts.indices)
for (i, src) in order.enumerated() {
    let dst = outDir.appendingPathComponent(String(format: "%06d.jpg", i))
    try? FileManager.default.removeItem(at: dst)
    try FileManager.default.moveItem(at: tmpURL(src), to: dst)
}
let ptsLines = order.map { String(format: "%.6f", pts[$0]) }
try ptsLines.joined(separator: "\n").write(to: outDir.appendingPathComponent("pts.txt"), atomically: true, encoding: .utf8)
try "\(srcSize.0) \(srcSize.1)".write(to: outDir.appendingPathComponent("source_size.txt"), atomically: true, encoding: .utf8)
print("decoded \(pts.count) frames from \(srcSize.0)x\(srcSize.1) to \(width)x\(height)"
      + (reordered ? " (reordered by presentation time)" : ""))
