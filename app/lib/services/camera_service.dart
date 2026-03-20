import 'dart:io';

import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

/// Image quality assessment result for the captured fundus image.
enum ImageQuality {
  good,
  blurry,
  tooDark,
  tooLight,
  unknown,
}

/// Result returned after capturing an image.
class CaptureResult {
  final String filePath;
  final ImageQuality quality;
  final int fileSizeBytes;

  const CaptureResult({
    required this.filePath,
    required this.quality,
    required this.fileSizeBytes,
  });
}

/// Manages camera access, image capture, and basic quality checks.
class CameraService {
  final ImagePicker _picker = ImagePicker();

  /// Request camera permission. Returns `true` if granted.
  Future<bool> requestCameraPermission() async {
    final status = await Permission.camera.request();
    return status.isGranted;
  }

  /// Capture a fundus image using the device camera.
  ///
  /// Returns `null` if the user cancels or permission is denied.
  Future<CaptureResult?> captureImage() async {
    final hasPermission = await requestCameraPermission();
    if (!hasPermission) return null;

    final XFile? photo = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 90,
      maxWidth: 2048,
      maxHeight: 2048,
      preferredCameraDevice: CameraDevice.rear,
    );

    if (photo == null) return null;

    // Copy to app-specific directory for persistence.
    final appDir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final savedPath = '${appDir.path}/captures/fundus_$timestamp.jpg';

    // Ensure captures directory exists.
    final capturesDir = Directory('${appDir.path}/captures');
    if (!await capturesDir.exists()) {
      await capturesDir.create(recursive: true);
    }

    final savedFile = await File(photo.path).copy(savedPath);
    final fileSize = await savedFile.length();

    // Basic quality heuristic based on file size.
    final quality = _assessQuality(fileSize);

    return CaptureResult(
      filePath: savedPath,
      quality: quality,
      fileSizeBytes: fileSize,
    );
  }

  /// Pick an image from the gallery (useful for testing / demo).
  Future<CaptureResult?> pickFromGallery() async {
    final XFile? photo = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
      maxWidth: 2048,
      maxHeight: 2048,
    );

    if (photo == null) return null;

    final appDir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final savedPath = '${appDir.path}/captures/fundus_$timestamp.jpg';

    final capturesDir = Directory('${appDir.path}/captures');
    if (!await capturesDir.exists()) {
      await capturesDir.create(recursive: true);
    }

    final savedFile = await File(photo.path).copy(savedPath);
    final fileSize = await savedFile.length();
    final quality = _assessQuality(fileSize);

    return CaptureResult(
      filePath: savedPath,
      quality: quality,
      fileSizeBytes: fileSize,
    );
  }

  /// Rudimentary quality assessment based on file size.
  ///
  /// In production, this would use the IQA model or histogram analysis.
  /// Very small files are likely blurry / out of focus; very large ones
  /// are likely fine.
  ImageQuality _assessQuality(int fileSize) {
    if (fileSize < 50 * 1024) {
      // < 50 KB — probably a bad capture
      return ImageQuality.tooDark;
    } else if (fileSize < 150 * 1024) {
      return ImageQuality.blurry;
    } else {
      return ImageQuality.good;
    }
  }

  /// Delete a previously captured file from local storage.
  Future<void> deleteCapture(String filePath) async {
    final file = File(filePath);
    if (await file.exists()) {
      await file.delete();
    }
  }
}
