import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/patient.dart';
import '../models/screening.dart';
import '../models/report.dart';
import '../services/api_service.dart';
import '../services/offline_queue.dart';
import 'connectivity_provider.dart';

/// State for the in-progress screening flow.
class ScreeningFlowState {
  final Screening? current;
  final Report? report;
  final List<Screening> recentScreenings;
  final bool isLoading;
  final String? error;
  final double uploadProgress;

  const ScreeningFlowState({
    this.current,
    this.report,
    this.recentScreenings = const [],
    this.isLoading = false,
    this.error,
    this.uploadProgress = 0.0,
  });

  ScreeningFlowState copyWith({
    Screening? current,
    Report? report,
    List<Screening>? recentScreenings,
    bool? isLoading,
    String? error,
    double? uploadProgress,
  }) {
    return ScreeningFlowState(
      current: current ?? this.current,
      report: report ?? this.report,
      recentScreenings: recentScreenings ?? this.recentScreenings,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      uploadProgress: uploadProgress ?? this.uploadProgress,
    );
  }
}

/// Manages the end-to-end screening workflow.
class ScreeningNotifier extends StateNotifier<ScreeningFlowState> {
  ScreeningNotifier(this._ref) : super(const ScreeningFlowState()) {
    loadRecentScreenings();
  }

  final Ref _ref;
  final _api = ApiService();
  final _offlineQueue = OfflineQueue();

  // ── Create Screening ─────────────────────────────────────────────────

  /// Start a new screening for the given patient.
  Future<String> createScreening(Patient patient) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final isOnline = _ref.read(isOnlineProvider);

      if (isOnline) {
        final screening = await _api.createScreening(patient);
        state = state.copyWith(current: screening, isLoading: false);
        return screening.id;
      } else {
        // Generate a temporary local ID and queue for later.
        final tempId =
            'local_${DateTime.now().millisecondsSinceEpoch}';
        final screening = Screening(
          id: tempId,
          patient: patient,
          status: ScreeningStatus.queuedOffline,
          createdAt: DateTime.now(),
        );
        await _offlineQueue.enqueue({
          'action': 'create_screening',
          'data': screening.toJson(),
        });
        state = state.copyWith(current: screening, isLoading: false);
        return tempId;
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Could not create screening: $e',
      );
      rethrow;
    }
  }

  // ── Image Capture ────────────────────────────────────────────────────

  /// Record a captured eye image locally and upload if online.
  Future<void> addEyeImage(Eye eye, String localPath) async {
    if (state.current == null) return;

    final image = EyeImage(eye: eye, localPath: localPath);
    final updatedImages = [
      ...state.current!.images.where((i) => i.eye != eye),
      image,
    ];

    state = state.copyWith(
      current: state.current!.copyWith(
        images: updatedImages,
        status: ScreeningStatus.capturing,
      ),
    );

    // Attempt upload immediately if online.
    final isOnline = _ref.read(isOnlineProvider);
    if (isOnline) {
      try {
        state = state.copyWith(uploadProgress: 0.0);
        final uploaded = await _api.uploadImage(
          state.current!.id,
          eye,
          localPath,
          onProgress: (progress) {
            state = state.copyWith(uploadProgress: progress);
          },
        );
        final newImages = updatedImages
            .map((i) => i.eye == eye ? uploaded : i)
            .toList();
        state = state.copyWith(
          current: state.current!.copyWith(images: newImages),
          uploadProgress: 1.0,
        );
      } catch (e) {
        // Queue for later upload.
        await _offlineQueue.enqueue({
          'action': 'upload_image',
          'screening_id': state.current!.id,
          'eye': eye == Eye.left ? 'left' : 'right',
          'local_path': localPath,
        });
      }
    } else {
      await _offlineQueue.enqueue({
        'action': 'upload_image',
        'screening_id': state.current!.id,
        'eye': eye == Eye.left ? 'left' : 'right',
        'local_path': localPath,
      });
    }
  }

  // ── Trigger Analysis ─────────────────────────────────────────────────

  /// Submit the screening for AI analysis and poll for results.
  Future<void> triggerAnalysis() async {
    if (state.current == null) return;
    state = state.copyWith(
      isLoading: true,
      error: null,
      current: state.current!.copyWith(status: ScreeningStatus.analyzing),
    );

    try {
      await _api.triggerAnalysis(state.current!.id);

      // Poll for results (simple approach — backend may also push via WS).
      Report? report;
      for (var attempt = 0; attempt < 30; attempt++) {
        await Future.delayed(const Duration(seconds: 2));
        report = await _api.getResults(state.current!.id);
        if (report != null) break;
      }

      if (report != null) {
        state = state.copyWith(
          current: state.current!.copyWith(
            status: ScreeningStatus.completed,
            resultId: report.id,
          ),
          report: report,
          isLoading: false,
        );
      } else {
        state = state.copyWith(
          isLoading: false,
          error: 'Analysis timed out. Please try again.',
        );
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Analysis failed: $e',
        current:
            state.current!.copyWith(status: ScreeningStatus.failed),
      );
    }
  }

  // ── Report ───────────────────────────────────────────────────────────

  /// Request a PDF report from the server.
  Future<String?> generateReport() async {
    if (state.current == null) return null;
    state = state.copyWith(isLoading: true);
    try {
      final url = await _api.generateReport(state.current!.id);
      state = state.copyWith(isLoading: false);
      return url;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Report generation failed: $e',
      );
      return null;
    }
  }

  // ── History ──────────────────────────────────────────────────────────

  Future<void> loadRecentScreenings() async {
    try {
      final isOnline = _ref.read(isOnlineProvider);
      if (isOnline) {
        final screenings = await _api.getRecentScreenings();
        state = state.copyWith(recentScreenings: screenings);
      }
    } catch (_) {
      // Silently fail — the list simply remains empty / stale.
    }
  }

  /// Reset flow state for a new screening.
  void reset() {
    state = state.copyWith(
      current: null,
      report: null,
      uploadProgress: 0.0,
      error: null,
    );
  }
}

/// Global screening flow provider.
final screeningProvider =
    StateNotifierProvider<ScreeningNotifier, ScreeningFlowState>(
  (ref) => ScreeningNotifier(ref),
);
