import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/screening.dart';
import '../providers/connectivity_provider.dart';
import '../config/app_config.dart';
import 'api_service.dart';
import 'offline_queue.dart';

/// Background service that drains the offline queue when the device
/// comes back online.
class SyncService {
  SyncService(this._ref) {
    _startPeriodicSync();
    // React immediately when connectivity changes.
    _ref.listen<NetworkStatus>(connectivityProvider, (prev, next) {
      if (next == NetworkStatus.online) {
        syncNow();
      }
    });
  }

  final Ref _ref;
  final _api = ApiService();
  final _queue = OfflineQueue();

  Timer? _timer;
  bool _syncing = false;

  /// Number of items successfully synced in the last run.
  int lastSyncCount = 0;

  void _startPeriodicSync() {
    _timer = Timer.periodic(
      const Duration(seconds: AppConfig.syncIntervalSeconds),
      (_) => syncNow(),
    );
  }

  /// Process the offline queue immediately.
  Future<void> syncNow() async {
    if (_syncing) return;
    if (_queue.isEmpty) return;
    if (_ref.read(connectivityProvider) == NetworkStatus.offline) return;

    _syncing = true;
    lastSyncCount = 0;

    try {
      while (!_queue.isEmpty) {
        final item = _queue.peek();
        if (item == null) break;

        final success = await _processItem(item);
        if (success) {
          await _queue.dequeue(); // Remove from head.
          lastSyncCount++;
        } else {
          // Increment retry counter; if too many retries, item is dropped.
          await _queue.incrementRetry(0);
          break; // Stop processing to avoid a tight loop on persistent errors.
        }
      }
    } finally {
      _syncing = false;
    }
  }

  /// Process a single queued item. Returns `true` on success.
  Future<bool> _processItem(Map<String, dynamic> item) async {
    try {
      final action = item['action'] as String;

      switch (action) {
        case 'create_screening':
          final data = item['data'] as Map<String, dynamic>;
          final screening = Screening.fromJson(data);
          await _api.createScreening(screening.patient);
          return true;

        case 'upload_image':
          final screeningId = item['screening_id'] as String;
          final eyeStr = item['eye'] as String;
          final localPath = item['local_path'] as String;
          final eye = eyeStr == 'left' ? Eye.left : Eye.right;
          await _api.uploadImage(screeningId, eye, localPath);
          return true;

        default:
          // Unknown action — discard it.
          return true;
      }
    } catch (_) {
      return false;
    }
  }

  /// Clean up the periodic timer.
  void dispose() {
    _timer?.cancel();
  }
}

/// Provider for the sync service — kept alive for the app's lifetime.
final syncServiceProvider = Provider<SyncService>((ref) {
  final service = SyncService(ref);
  ref.onDispose(service.dispose);
  return service;
});
