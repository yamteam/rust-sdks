/*
 * Copyright 2026 LiveKit, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "api/video/encoded_image.h"
#include "api/video/video_frame_buffer.h"

namespace livekit {

enum class EncodedVideoCodec {
  kH264,
  kH265,
  kVP8,
  kVP9,
  kAV1,
};

enum class EncodedFrameType {
  kKey,
  kDelta,
};

struct EncodedRateControlRequest {
  bool has_request = false;
  uint64_t target_bitrate_bps = 0;
  double framerate_fps = 0.0;
};

// Simulcast layers a pre-encoded source can carry; matches libwebrtc's
// kMaxSimulcastStreams.
constexpr size_t kMaxEncodedLayers = 3;

// Latest-wins rate-control mailbox shared between the pass-through encoder and
// the Rust capture side.
class EncodedRateControlState {
 public:
  void Store(uint64_t target_bitrate_bps, double framerate_fps);
  EncodedRateControlRequest Take();

 private:
  std::mutex mutex_;
  EncodedRateControlRequest request_;
};

// Signals from the pass-through encoders back to the capture side, one slot
// per simulcast layer (a single-layer source uses slot 0).
class EncodedLayerSignals {
 public:
  // A keyframe is needed on `layer` (PLI/FIR, a gap, the layer resumed).
  void RequestKeyframe(size_t layer);
  bool TakeKeyframeRequest(size_t layer);

  void StoreRate(size_t layer, uint64_t target_bitrate_bps,
                 double framerate_fps);
  EncodedRateControlRequest TakeRate(size_t layer);

  // The encoder of `layer` was asked to encode a frame. libwebrtc does not
  // call the encoder of a paused layer (no subscriber, no bandwidth), so the
  // capture side can stop producing a layer nobody asks for.
  void MarkWanted(size_t layer);
  // Milliseconds since `layer` was last asked for, or -1 if never.
  int64_t MillisSinceWanted(size_t layer) const;

 private:
  std::array<std::atomic<bool>, kMaxEncodedLayers> keyframe_{};
  std::array<EncodedRateControlState, kMaxEncodedLayers> rates_;
  std::array<std::atomic<int64_t>, kMaxEncodedLayers> wanted_ms_{};
};

// One simulcast layer of an encoded frame. An empty payload means the capture
// side did not encode this layer for this frame.
struct EncodedLayer {
  int width = 0;
  int height = 0;
  EncodedFrameType frame_type = EncodedFrameType::kDelta;
  webrtc::scoped_refptr<webrtc::EncodedImageBuffer> payload;
  uint64_t sequence = 0;
};

// A native WebRTC frame buffer carrying one encoded video access unit per
// simulcast layer, lowest layer first. libwebrtc's simulcast adapter hands
// native buffers to every layer's encoder unscaled, and each pass-through
// encoder forwards the layer that matches its resolution.
class EncodedVideoFrameBuffer : public webrtc::VideoFrameBuffer {
 public:
  // `signals` is shared with the owning video source: the pass-through
  // encoder raises keyframe requests and rate targets there, and the capture
  // side polls them to drive the upstream encoders.
  EncodedVideoFrameBuffer(
      int width,
      int height,
      EncodedVideoCodec codec,
      EncodedFrameType frame_type,
      webrtc::scoped_refptr<webrtc::EncodedImageBuffer> payload,
      std::shared_ptr<EncodedLayerSignals> signals = nullptr,
      uint64_t sequence = 0);
  EncodedVideoFrameBuffer(EncodedVideoCodec codec,
                          std::vector<EncodedLayer> layers,
                          std::shared_ptr<EncodedLayerSignals> signals);
  ~EncodedVideoFrameBuffer() override = default;

  Type type() const override;
  // The size of the top layer: the frame libwebrtc configures streams for.
  int width() const override;
  int height() const override;
  webrtc::scoped_refptr<webrtc::I420BufferInterface> ToI420() override;
  webrtc::scoped_refptr<webrtc::VideoFrameBuffer> CropAndScale(
      int offset_x,
      int offset_y,
      int crop_width,
      int crop_height,
      int scaled_width,
      int scaled_height) override;

  EncodedVideoCodec codec() const { return codec_; }
  const std::vector<EncodedLayer>& layers() const { return layers_; }
  // The layer an encoder configured for `width` x `height` sends: the one
  // closest in area.
  size_t LayerIndexFor(int width, int height) const;

  void request_keyframe(size_t layer) const;
  void set_rate_control_request(size_t layer,
                                uint64_t target_bitrate_bps,
                                double framerate_fps) const;
  void mark_wanted(size_t layer) const;

  static EncodedVideoFrameBuffer* FromNative(webrtc::VideoFrameBuffer* buffer);

 private:
  EncodedVideoCodec codec_;
  std::vector<EncodedLayer> layers_;
  std::shared_ptr<EncodedLayerSignals> signals_;
};

}  // namespace livekit
