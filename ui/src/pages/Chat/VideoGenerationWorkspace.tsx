/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import {
  ChangeEvent,
  CSSProperties,
  FC,
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';

import classNames from 'classnames';
import { v4 as uuidv4 } from 'uuid';

import { Icon } from '@/components';
import { LOGGED_TOKEN_STORAGE_KEY } from '@/common/constants';
import type {
  AiSubscriptionOverview,
  AiVideoGeneration,
  AiVideoModel,
} from '@/common/interface';
import {
  generateAiVideo,
  getAiVideoGenerations,
  getAiVideoModels,
} from '@/services/client/ai';
import Storage from '@/utils/storage';
import './imageGeneration.scss';

interface ReferenceImage {
  id: string;
  name: string;
  url: string;
}

type TaskStatus = 'queued' | 'generating' | 'completed' | 'failed';

interface VideoTask {
  id: string;
  prompt: string;
  model: string;
  siteModelID: string;
  size: string;
  ratio: string;
  quality: string;
  seconds: number;
  preset: string;
  status: TaskStatus;
  createdAt: number;
  progress: number;
  videoURL: string;
  referenceImages: string[];
  error?: string;
}

interface PreviewVideo {
  src: string;
  rawURL: string;
  task: VideoTask;
}

interface IProps {
  subscription: AiSubscriptionOverview | null;
  onRefreshSubscription: () => void;
  onOpenSubscription: () => void;
}

const sizeOptions = [
  { value: 'auto', label: '自动', meta: '模型默认', ratio: 'auto' },
  { value: '1280x720', label: '16:9', meta: '1280 × 720', ratio: '16:9' },
  { value: '720x1280', label: '9:16', meta: '720 × 1280', ratio: '9:16' },
  { value: '1024x1024', label: '1:1', meta: '1024 × 1024', ratio: '1:1' },
  { value: '1792x1024', label: '16:9+', meta: '1792 × 1024', ratio: '16:9' },
  { value: '1024x1792', label: '9:16+', meta: '1024 × 1792', ratio: '9:16' },
];

const secondOptions = [
  { value: 6, label: '6 秒' },
  { value: 10, label: '10 秒' },
  { value: 12, label: '12 秒' },
  { value: 16, label: '16 秒' },
  { value: 20, label: '20 秒' },
];

const qualityOptions = [
  { value: '720p', label: '高清' },
  { value: '480p', label: '标准' },
];

const presetOptions = [
  { value: 'normal', label: '标准' },
  { value: 'fun', label: '创意' },
  { value: 'custom', label: '自定义' },
  { value: 'spicy', label: '强烈' },
];

const readFileAsDataURL = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

const formatQuota = (value?: number) => {
  if (value === -1) {
    return '无限制';
  }
  return Number(value || 0).toLocaleString();
};

const normalizeVideoGenerationError = (err: any) => {
  const rawMessage =
    typeof err?.msg === 'string'
      ? err.msg
      : err instanceof Error
        ? err.message
        : '';
  if (rawMessage.includes('today video quota is insufficient')) {
    return '今日视频额度不足，请升级或兑换订阅';
  }
  if (rawMessage.includes('monthly video quota is insufficient')) {
    return '本月视频额度不足，请升级或兑换订阅';
  }
  return rawMessage || '视频生成失败，请检查模型配置或稍后重试';
};

const getModelName = (model?: AiVideoModel) => {
  if (!model) {
    return '选择模型';
  }
  return model.display_name || model.site_model_id;
};

const normalizeTaskStatus = (status?: string): TaskStatus => {
  if (status === 'queued' || status === 'generating' || status === 'failed') {
    return status;
  }
  if (status === 'in_progress' || status === 'processing') {
    return 'generating';
  }
  return 'completed';
};

const getTaskStatusLabel = (status: TaskStatus) => {
  if (status === 'queued') {
    return '排队中';
  }
  if (status === 'generating') {
    return '生成中';
  }
  if (status === 'failed') {
    return '失败';
  }
  return '已完成';
};

const getSizeRatio = (size?: string, fallbackRatio = '16:9') => {
  const sizeMatch = size?.match(/^(\d+)x(\d+)$/);
  if (sizeMatch) {
    return `${sizeMatch[1]} / ${sizeMatch[2]}`;
  }
  const ratioMatch = fallbackRatio.match(/^(\d+):(\d+)$/);
  if (ratioMatch) {
    return `${ratioMatch[1]} / ${ratioMatch[2]}`;
  }
  return '16 / 9';
};

const getSizeRatioParts = (size?: string, fallbackRatio = '16:9') => {
  const sizeMatch = size?.match(/^(\d+)x(\d+)$/);
  if (sizeMatch) {
    return {
      width: Number(sizeMatch[1]) || 16,
      height: Number(sizeMatch[2]) || 9,
    };
  }
  const ratioMatch = fallbackRatio.match(/^(\d+):(\d+)$/);
  if (ratioMatch) {
    return {
      width: Number(ratioMatch[1]) || 16,
      height: Number(ratioMatch[2]) || 9,
    };
  }
  return { width: 16, height: 9 };
};

const getTaskRetryKey = (
  task: Pick<
    VideoTask,
    'prompt' | 'siteModelID' | 'size' | 'quality' | 'seconds' | 'preset'
  >,
) =>
  [
    task.prompt,
    task.siteModelID,
    task.size,
    task.quality,
    task.seconds,
    task.preset,
  ].join('|');

const removeRetrySupersededFailedTasks = (tasks: VideoTask[]) => {
  const successfulRetryKeys = new Set(
    tasks
      .filter((task) => task.status !== 'failed')
      .map((task) => getTaskRetryKey(task)),
  );
  return tasks.filter(
    (task) =>
      task.status !== 'failed' ||
      !successfulRetryKeys.has(getTaskRetryKey(task)),
  );
};

const mapGenerationToTask = (item: AiVideoGeneration): VideoTask => {
  const taskStatus = normalizeTaskStatus(item.status);
  const ratio =
    item.aspect_ratio ||
    sizeOptions.find((option) => option.value === item.size)?.ratio ||
    '16:9';
  return {
    id: item.generation_id,
    prompt: item.prompt,
    model: item.site_model_id,
    siteModelID: item.site_model_id,
    size: item.size || '1280x720',
    ratio,
    quality: item.quality || '720p',
    seconds: item.seconds || 6,
    preset: item.preset || 'normal',
    status: taskStatus,
    createdAt: item.created_at * 1000,
    progress:
      taskStatus === 'completed' || taskStatus === 'failed'
        ? 100
        : Math.max(18, item.progress || 28),
    videoURL: item.video_url || '',
    referenceImages: item.reference_images || [],
    error: item.error || '',
  };
};

const resolveVideoURL = (url: string) => {
  if (!url || url.startsWith('blob:') || url.startsWith('data:')) {
    return url;
  }
  const appendAssetToken = (assetURL: string) => {
    const token = Storage.get(LOGGED_TOKEN_STORAGE_KEY);
    if (!token) {
      return assetURL;
    }
    try {
      const parsed = new URL(assetURL, window.location.origin);
      parsed.searchParams.set('Authorization', token);
      return parsed.pathname + parsed.search + parsed.hash;
    } catch {
      return assetURL;
    }
  };
  try {
    const parsed = new URL(url, window.location.origin);
    const match = parsed.pathname.match(
      /^\/uploads\/ai-videos\/([^/]+)\/([^/?#]+)$/,
    );
    if (match) {
      return appendAssetToken(
        `/answer/api/v1/ai-video/assets/${encodeURIComponent(
          match[1],
        )}/${encodeURIComponent(match[2])}`,
      );
    }
  } catch {
    // Keep the original URL if parsing fails.
  }
  if (/^https?:\/\//i.test(url)) {
    return url;
  }
  const apiBase = process.env.REACT_APP_API_URL || '';
  if (apiBase && apiBase !== '/' && url.startsWith('/')) {
    return `${apiBase.replace(/\/$/, '')}${url}`;
  }
  return url;
};

const getDownloadFilename = (task?: VideoTask) => {
  if (!task) {
    return 'ai-video.mp4';
  }
  return `${task.id || 'ai-video'}.mp4`;
};

const getVideoPreviewStyle = (task: Pick<VideoTask, 'size' | 'ratio'>) => {
  const ratioParts = getSizeRatioParts(task.size, task.ratio);
  return {
    '--hcai-preview-ratio': getSizeRatio(task.size, task.ratio),
    '--hcai-preview-ratio-width': ratioParts.width,
    '--hcai-preview-ratio-height': ratioParts.height,
  } as CSSProperties;
};

const VideoGenerationWorkspace: FC<IProps> = ({
  subscription,
  onRefreshSubscription,
  onOpenSubscription,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeVideoRef = useRef<HTMLVideoElement | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const [models, setModels] = useState<AiVideoModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [selectedModelID, setSelectedModelID] = useState('');
  const [prompt, setPrompt] = useState('');
  const [size, setSize] = useState('1280x720');
  const [seconds, setSeconds] = useState(6);
  const [quality, setQuality] = useState('720p');
  const [preset, setPreset] = useState('normal');
  const [referenceImages, setReferenceImages] = useState<ReferenceImage[]>([]);
  const [tasks, setTasks] = useState<VideoTask[]>([]);
  const [sidebarTaskSearch, setSidebarTaskSearch] = useState('');
  const [mobileTaskSearch, setMobileTaskSearch] = useState('');
  const [activeTaskID, setActiveTaskID] = useState('');
  const [error, setError] = useState('');
  const [actionNotice, setActionNotice] = useState('');
  const [generating, setGenerating] = useState(false);
  const [sidebarTaskPortalHost, setSidebarTaskPortalHost] =
    useState<Element | null>(null);
  const [mobileSideNavTaskPortalHost, setMobileSideNavTaskPortalHost] =
    useState<Element | null>(null);
  const [previewVideo, setPreviewVideo] = useState<PreviewVideo | null>(null);
  const [activeVideoPlaying, setActiveVideoPlaying] = useState(false);
  const [previewVideoPlaying, setPreviewVideoPlaying] = useState(false);
  const [videoBlobURLs, setVideoBlobURLs] = useState<Record<string, string>>(
    {},
  );
  const videoBlobURLsRef = useRef<Record<string, string>>({});
  const loadingVideoURLsRef = useRef<Set<string>>(new Set());

  const selectedModel = useMemo(
    () => models.find((model) => model.site_model_id === selectedModelID),
    [models, selectedModelID],
  );
  const selectedSize =
    size === 'auto' ? selectedModel?.default_size || '1280x720' : size;
  const selectedRatio =
    sizeOptions.find((option) => option.value === size)?.ratio || '16:9';
  const activeTask = tasks.find((task) => task.id === activeTaskID) || tasks[0];
  const quotaRemaining = subscription?.video_quota_remaining ?? 0;
  const dailyRemaining = subscription?.video_daily_remaining ?? 0;
  const canGenerate = Boolean(prompt.trim()) && Boolean(selectedModelID);
  const activeVideoURL = activeTask?.videoURL
    ? videoBlobURLs[activeTask.videoURL] || resolveVideoURL(activeTask.videoURL)
    : '';
  const shouldPollVideoTasks = tasks.some(
    (task) =>
      task.status === 'queued' ||
      task.status === 'generating' ||
      (task.status === 'completed' &&
        (!task.videoURL ||
          (!/^https?:\/\//i.test(task.videoURL) &&
            !videoBlobURLs[task.videoURL]))),
  );
  const videoURLKey = useMemo(
    () => tasks.map((task) => task.videoURL).join('|'),
    [tasks],
  );
  const getFilteredTasks = useCallback(
    (search: string) => {
      const query = search.trim().toLocaleLowerCase();
      if (!query) {
        return tasks;
      }
      return tasks.filter((task) =>
        [
          task.prompt,
          task.model,
          task.siteModelID,
          task.size,
          task.ratio,
          task.quality,
          `${task.seconds} 秒`,
          getTaskStatusLabel(task.status),
          task.error || '',
        ]
          .filter(Boolean)
          .join('\n')
          .toLocaleLowerCase()
          .includes(query),
      );
    },
    [tasks],
  );

  const refreshVideoGenerations = useCallback(async () => {
    const resp = await getAiVideoGenerations();
    const historyTasks = removeRetrySupersededFailedTasks(
      (resp || []).map(mapGenerationToTask),
    );
    setTasks((prev) => {
      const historyTaskIDs = new Set(historyTasks.map((task) => task.id));
      const localPendingTasks = prev.filter(
        (task) =>
          (task.status === 'queued' || task.status === 'generating') &&
          !historyTaskIDs.has(task.id),
      );
      return removeRetrySupersededFailedTasks([
        ...localPendingTasks,
        ...historyTasks,
      ]);
    });
    setActiveTaskID((prev) => {
      if (prev && historyTasks.some((task) => task.id === prev)) {
        return prev;
      }
      return historyTasks[0]?.id || '';
    });
  }, []);

  const showActionNotice = (message: string) => {
    setActionNotice(message);
    window.setTimeout(() => {
      setActionNotice('');
    }, 1800);
  };

  useEffect(() => {
    let mounted = true;
    setModelsLoading(true);
    getAiVideoModels()
      .then((resp) => {
        if (!mounted) {
          return;
        }
        const nextModels = resp || [];
        setModels(nextModels);
        if (nextModels[0]) {
          setSelectedModelID(nextModels[0].site_model_id);
          setSize(nextModels[0].default_size || '1280x720');
          setSeconds(nextModels[0].default_seconds || 6);
          setQuality(nextModels[0].default_resolution || '720p');
          setPreset(nextModels[0].default_preset || 'normal');
        }
      })
      .catch(() => {
        if (mounted) {
          setError('视频模型加载失败，请联系管理员检查配置');
        }
      })
      .finally(() => {
        if (mounted) {
          setModelsLoading(false);
        }
      });

    getAiVideoGenerations()
      .then((resp) => {
        if (!mounted || !resp?.length) {
          return;
        }
        const historyTasks = removeRetrySupersededFailedTasks(
          resp.map(mapGenerationToTask),
        );
        setTasks(historyTasks);
        setActiveTaskID(historyTasks[0].id);
      })
      .catch(() => undefined);

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (generating || !shouldPollVideoTasks) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      refreshVideoGenerations()
        .then(() => {
          onRefreshSubscription();
        })
        .catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [
    generating,
    onRefreshSubscription,
    refreshVideoGenerations,
    shouldPollVideoTasks,
  ]);

  useEffect(() => {
    videoBlobURLsRef.current = videoBlobURLs;
  }, [videoBlobURLs]);

  useEffect(() => {
    setActiveVideoPlaying(false);
  }, [activeTask?.id, activeVideoURL]);

  useEffect(() => {
    if (!previewVideo) {
      return undefined;
    }
    setPreviewVideoPlaying(false);
    const handleKeyDown = (evt: KeyboardEvent) => {
      if (evt.key === 'Escape') {
        setPreviewVideo(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [previewVideo]);

  useEffect(() => {
    setSidebarTaskPortalHost(
      document.querySelector('#hcai-sidebar-video-tasks'),
    );
  }, []);

  useEffect(() => {
    let frame = 0;
    let cancelled = false;

    const resolveMobileSideNavTaskHost = () => {
      if (cancelled) return;
      const host = document.querySelector('#hcai-mobile-sidenav-video-tasks');
      setMobileSideNavTaskPortalHost(host);
      if (!host) {
        frame = window.requestAnimationFrame(resolveMobileSideNavTaskHost);
      }
    };

    resolveMobileSideNavTaskHost();
    return () => {
      cancelled = true;
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  useEffect(() => {
    return () => {
      Array.from(new Set(Object.values(videoBlobURLsRef.current))).forEach(
        (objectURL) => {
          URL.revokeObjectURL(objectURL);
        },
      );
    };
  }, []);

  useEffect(() => {
    const token = Storage.get(LOGGED_TOKEN_STORAGE_KEY) || '';
    const videoURLs = Array.from(new Set(tasks.map((task) => task.videoURL)))
      .filter(Boolean)
      .filter((url) => !url.startsWith('http'));

    videoURLs.forEach((rawURL) => {
      if (
        rawURL.startsWith('data:') ||
        videoBlobURLsRef.current[rawURL] ||
        loadingVideoURLsRef.current.has(rawURL)
      ) {
        return;
      }
      const requestURL = resolveVideoURL(rawURL);
      if (!requestURL || requestURL.startsWith('data:')) {
        return;
      }
      const headers: Record<string, string> = {};
      if (token) {
        headers.Authorization = token;
      }
      loadingVideoURLsRef.current.add(rawURL);
      fetch(requestURL, { credentials: 'include', headers })
        .then((resp) => {
          if (!resp.ok) {
            throw new Error(`Video load failed: ${resp.status}`);
          }
          return resp.blob();
        })
        .then((blob) => {
          const objectURL = URL.createObjectURL(blob);
          setVideoBlobURLs((prev) => {
            if (prev[rawURL]) {
              URL.revokeObjectURL(objectURL);
              return prev;
            }
            const next = { ...prev, [rawURL]: objectURL };
            videoBlobURLsRef.current = next;
            return next;
          });
        })
        .catch(() => undefined)
        .finally(() => {
          loadingVideoURLsRef.current.delete(rawURL);
        });
    });
  }, [videoURLKey, tasks]);

  const addReferenceImages = async (files: File[]) => {
    const imageFiles = files.filter((file) => file.type.startsWith('image/'));
    if (imageFiles.length === 0) {
      return;
    }
    if (referenceImages.length + imageFiles.length > 4) {
      setError('最多添加 4 张参考图');
      return;
    }
    const oversized = imageFiles.find((file) => file.size > 5 * 1024 * 1024);
    if (oversized) {
      setError('单张参考图不能超过 5MB');
      return;
    }
    const images = await Promise.all(
      imageFiles.map(async (file) => ({
        id: `${file.name}-${file.lastModified}-${Math.random()
          .toString(36)
          .slice(2)}`,
        name: file.name,
        url: await readFileAsDataURL(file),
      })),
    );
    setReferenceImages((prev) => [...prev, ...images]);
    setError('');
  };

  const handleReferenceSelect = (evt: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(evt.target.files || []);
    evt.target.value = '';
    addReferenceImages(files).catch(() => {
      setError('参考图读取失败，请重新选择');
    });
  };

  const removeReferenceImage = (id: string) => {
    setReferenceImages((prev) => prev.filter((image) => image.id !== id));
  };

  const downloadVideo = async (videoURL: string, task?: VideoTask) => {
    if (!videoURL) {
      showActionNotice('暂无可下载视频');
      return;
    }
    try {
      const token = Storage.get(LOGGED_TOKEN_STORAGE_KEY) || '';
      const headers: Record<string, string> = {};
      if (token) {
        headers.Authorization = token;
      }
      const resp = await fetch(videoURL, { credentials: 'include', headers });
      if (!resp.ok) {
        throw new Error(`Download failed: ${resp.status}`);
      }
      const blob = await resp.blob();
      const objectURL = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectURL;
      link.download = getDownloadFilename(task);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectURL), 0);
      showActionNotice('已开始下载');
    } catch {
      showActionNotice('下载失败，请稍后重试');
    }
  };

  const openVideoPreview = (task: VideoTask, src: string, rawURL: string) => {
    activeVideoRef.current?.pause();
    setActiveVideoPlaying(false);
    setPreviewVideo({ task, src, rawURL });
  };

  const playActiveVideo = () => {
    activeVideoRef.current?.play().catch(() => undefined);
  };

  const playPreviewVideo = () => {
    previewVideoRef.current?.play().catch(() => undefined);
  };

  const runVideoTask = async (task: VideoTask) => {
    if (dailyRemaining !== -1 && dailyRemaining < 1) {
      setError('今日视频额度不足，请升级或兑换订阅');
      return;
    }
    if (quotaRemaining !== -1 && quotaRemaining < 1) {
      setError('本月视频额度不足，请升级或兑换订阅');
      return;
    }

    setTasks((prev) =>
      prev.map((item) =>
        item.id === task.id
          ? {
              ...item,
              status: 'generating',
              progress: 0,
              videoURL: '',
              error: '',
            }
          : item,
      ),
    );
    setActiveTaskID(task.id);
    setError('');
    setGenerating(true);
    try {
      const resp = await generateAiVideo({
        prompt: task.prompt,
        model: task.siteModelID,
        size: task.size,
        quality: task.quality,
        seconds: task.seconds,
        preset: task.preset,
        reference_images: task.referenceImages,
      });
      const nextTaskID = resp.generation_id || task.id;
      const nextStatus = normalizeTaskStatus(resp.status);
      setTasks((prev) =>
        removeRetrySupersededFailedTasks(
          prev.map((item) =>
            item.id === task.id
              ? {
                  ...item,
                  id: nextTaskID,
                  size: resp.size || item.size,
                  seconds: resp.seconds || item.seconds,
                  status: nextStatus,
                  progress: resp.progress ?? item.progress,
                }
              : item,
          ),
        ),
      );
      setActiveTaskID(nextTaskID);
      refreshVideoGenerations().catch(() => undefined);
      onRefreshSubscription();
    } catch (err: any) {
      const errorMessage = normalizeVideoGenerationError(err);
      setTasks((prev) =>
        prev.map((item) =>
          item.id === task.id
            ? {
                ...item,
                status: 'failed',
                progress: 100,
                error: errorMessage,
              }
            : item,
        ),
      );
      setError(errorMessage);
    } finally {
      setGenerating(false);
    }
  };

  const retryTask = (task: VideoTask) => {
    if (generating || task.status !== 'failed') {
      return;
    }
    runVideoTask(task);
  };

  const selectTask = (taskID: string, closePanel = false) => {
    setActiveTaskID(taskID);
    if (closePanel) {
      window.dispatchEvent(new CustomEvent('hcai-close-mobile-side-nav'));
    }
  };

  const submitGeneration = async (evt: FormEvent) => {
    evt.preventDefault();
    if (!canGenerate) {
      setError(!selectedModelID ? '请先选择视频模型' : '请输入视频提示词');
      return;
    }
    if (dailyRemaining !== -1 && dailyRemaining < 1) {
      setError('今日视频额度不足，请升级或兑换订阅');
      return;
    }
    if (quotaRemaining !== -1 && quotaRemaining < 1) {
      setError('本月视频额度不足，请升级或兑换订阅');
      return;
    }

    const taskID = uuidv4();
    const nextTask: VideoTask = {
      id: taskID,
      prompt: prompt.trim(),
      model: getModelName(selectedModel),
      siteModelID: selectedModelID,
      size: selectedSize,
      ratio: selectedRatio === 'auto' ? '16:9' : selectedRatio,
      quality,
      seconds,
      preset,
      status: 'generating',
      createdAt: Date.now(),
      progress: 0,
      videoURL: '',
      referenceImages: referenceImages.map((image) => image.url),
      error: '',
    };
    setTasks((prev) => [nextTask, ...prev]);
    setActiveTaskID(taskID);
    runVideoTask(nextTask).then(() => {
      setPrompt('');
    });
  };

  const renderTaskItem = (
    task: VideoTask,
    closePanel = false,
    showRetry = true,
  ) => (
    <div
      className={classNames('hcai-task-item', {
        active: task.id === activeTaskID,
        'has-retry': showRetry && task.status === 'failed',
      })}
      key={task.id}>
      <button
        type="button"
        className="hcai-task-select"
        onClick={() => selectTask(task.id, closePanel)}>
        <span className={`hcai-task-dot ${task.status}`} />
        <div className="hcai-task-body">
          <strong>{task.prompt}</strong>
          <span>
            {task.status === 'failed' && task.error
              ? task.error
              : `${task.model} · ${task.size} · ${task.seconds} 秒`}
          </span>
        </div>
        <em>{getTaskStatusLabel(task.status)}</em>
      </button>
      {showRetry && task.status === 'failed' ? (
        <button
          type="button"
          className="hcai-task-retry"
          disabled={generating}
          onClick={() => retryTask(task)}>
          <Icon name="arrow-clockwise" />
          <span>重试</span>
        </button>
      ) : null}
    </div>
  );

  const renderTaskPanel = ({
    closePanel = false,
    showRetry = true,
    search,
    onSearchChange,
  }: {
    closePanel?: boolean;
    showRetry?: boolean;
    search?: string;
    onSearchChange?: (value: string) => void;
  }) => {
    const canSearch = search !== undefined && onSearchChange !== undefined;
    const panelTasks = canSearch ? getFilteredTasks(search) : tasks;
    const trimmedSearch = canSearch ? search.trim() : '';
    return (
      <div className="hcai-task-panel hcai-video-task-panel">
        <div className="hcai-task-head">
          <span>任务队列</span>
          <strong>
            {trimmedSearch
              ? `${panelTasks.length}/${tasks.length}`
              : tasks.length}
          </strong>
        </div>
        {canSearch ? (
          <div className="hcai-agent-conversation-search-wrap">
            <svg
              aria-hidden="true"
              className="hcai-agent-conversation-search-icon"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="search"
              value={search}
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder="搜索任务..."
              className="hcai-agent-conversation-search"
            />
          </div>
        ) : null}
        <div className="hcai-task-list">
          {panelTasks.length > 0 ? (
            panelTasks.map((task) =>
              renderTaskItem(task, closePanel, showRetry),
            )
          ) : (
            <span className="hcai-task-empty">
              {tasks.length > 0 ? '没有匹配的任务' : '暂无任务'}
            </span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="hcai-image-workspace hcai-video-workspace">
      {sidebarTaskPortalHost
        ? createPortal(
            renderTaskPanel({
              closePanel: false,
              showRetry: false,
              search: sidebarTaskSearch,
              onSearchChange: setSidebarTaskSearch,
            }),
            sidebarTaskPortalHost,
          )
        : null}
      {mobileSideNavTaskPortalHost
        ? createPortal(
            renderTaskPanel({
              closePanel: true,
              search: mobileTaskSearch,
              onSearchChange: setMobileTaskSearch,
            }),
            mobileSideNavTaskPortalHost,
          )
        : null}
      <section className="hcai-image-composer">
        <div className="hcai-image-head">
          <div>
            <span className="hcai-image-kicker">HCAI Video</span>
            <h1>视频生成</h1>
          </div>
          <button type="button" onClick={onOpenSubscription}>
            <Icon name="camera-reels" />
            <span>
              今日 {formatQuota(dailyRemaining)} · 本月{' '}
              {formatQuota(quotaRemaining)}
            </span>
          </button>
        </div>

        <form className="hcai-image-form" onSubmit={submitGeneration}>
          <div className="hcai-image-field">
            <label htmlFor="hcai-video-prompt">提示词</label>
            <textarea
              id="hcai-video-prompt"
              value={prompt}
              rows={6}
              placeholder="描述你想生成的视频画面、镜头运动、主体动作和整体风格"
              onChange={(evt) => setPrompt(evt.target.value)}
            />
          </div>

          <div className="hcai-image-panel">
            <div className="hcai-image-panel-title">
              <Icon name="image" />
              <span>参考图</span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hcai-image-input"
              onChange={handleReferenceSelect}
            />
            <div className="hcai-reference-grid">
              {referenceImages.map((image) => (
                <div className="hcai-reference-image" key={image.id}>
                  <img src={image.url} alt={image.name} />
                  <button
                    type="button"
                    aria-label="移除参考图"
                    onClick={() => removeReferenceImage(image.id)}>
                    <Icon name="x" />
                  </button>
                </div>
              ))}
              <button
                type="button"
                className="hcai-reference-add"
                onClick={() => fileInputRef.current?.click()}>
                <Icon name="plus-lg" />
                <span>添加</span>
              </button>
            </div>
          </div>

          <div className="hcai-image-field compact">
            <label htmlFor="hcai-video-model">模型</label>
            <select
              id="hcai-video-model"
              value={selectedModelID}
              disabled={modelsLoading || models.length === 0}
              onChange={(evt) => {
                const nextModelID = evt.target.value;
                const nextModel = models.find(
                  (model) => model.site_model_id === nextModelID,
                );
                setSelectedModelID(nextModelID);
                if (nextModel) {
                  setSize(nextModel.default_size || '1280x720');
                  setSeconds(nextModel.default_seconds || 6);
                  setQuality(nextModel.default_resolution || '720p');
                  setPreset(nextModel.default_preset || 'normal');
                }
              }}>
              {modelsLoading ? <option>加载模型...</option> : null}
              {models.map((model) => (
                <option value={model.site_model_id} key={model.site_model_id}>
                  {getModelName(model)}
                </option>
              ))}
            </select>
          </div>

          <div className="hcai-image-section-label">时长</div>
          <div className="hcai-style-options hcai-video-choice-row">
            {secondOptions.map((option) => (
              <button
                type="button"
                className={seconds === option.value ? 'active' : ''}
                key={option.value}
                onClick={() => setSeconds(option.value)}>
                {option.label}
              </button>
            ))}
          </div>

          <div className="hcai-image-section-label">比例</div>
          <div className="hcai-ratio-options hcai-video-size-options">
            {sizeOptions.map((option) => (
              <button
                type="button"
                className={size === option.value ? 'active' : ''}
                key={option.value}
                onClick={() => setSize(option.value)}>
                <strong>{option.label}</strong>
                <span>{option.meta}</span>
              </button>
            ))}
          </div>

          <div className="hcai-image-section-label">质量</div>
          <div className="hcai-style-options hcai-video-choice-row">
            {qualityOptions.map((option) => (
              <button
                type="button"
                className={quality === option.value ? 'active' : ''}
                key={option.value}
                onClick={() => setQuality(option.value)}>
                {option.label}
              </button>
            ))}
          </div>

          <div className="hcai-image-section-label">风格</div>
          <div className="hcai-style-options hcai-video-choice-row">
            {presetOptions.map((option) => (
              <button
                type="button"
                className={preset === option.value ? 'active' : ''}
                key={option.value}
                onClick={() => setPreset(option.value)}>
                {option.label}
              </button>
            ))}
          </div>

          {error ? <div className="hcai-image-error">{error}</div> : null}

          <button
            type="submit"
            className="hcai-image-generate"
            disabled={!canGenerate || generating}>
            <Icon name="camera-reels" />
            <span>{generating ? '生成中' : '生成视频'}</span>
          </button>
        </form>
      </section>

      <section className="hcai-image-preview">
        <div className="hcai-preview-toolbar">
          <div>
            <span>预览</span>
            <strong>
              {activeTask ? getTaskStatusLabel(activeTask.status) : '待生成'}
            </strong>
          </div>
          <div className="hcai-preview-actions">
            {actionNotice ? (
              <span className="hcai-preview-notice">{actionNotice}</span>
            ) : null}
            <button
              type="button"
              title="下载"
              disabled={!activeVideoURL}
              onClick={() => downloadVideo(activeVideoURL, activeTask)}>
              <Icon name="download" />
            </button>
          </div>
        </div>

        <div className="hcai-preview-canvas">
          {activeTask ? (
            <div
              className={classNames('hcai-preview-result', {
                loading: activeTask.status === 'generating',
              })}>
              <div
                className={classNames(
                  'hcai-preview-tile',
                  'hcai-video-preview-tile',
                  {
                    'has-image': Boolean(activeTask.videoURL),
                  },
                )}
                style={getVideoPreviewStyle(activeTask)}>
                {activeTask.status === 'generating' ||
                activeTask.status === 'queued' ? (
                  <>
                    <Icon name="camera-reels" />
                    <span>{activeTask.ratio}</span>
                    <div className="hcai-preview-progress">
                      <span style={{ width: `${activeTask.progress}%` }} />
                    </div>
                  </>
                ) : activeTask.videoURL ? (
                  activeVideoURL ? (
                    <>
                      <div className="hcai-preview-video-shell">
                        <video
                          ref={activeVideoRef}
                          className="hcai-preview-video"
                          src={activeVideoURL}
                          playsInline
                          onPlay={() => setActiveVideoPlaying(true)}
                          onPause={() => setActiveVideoPlaying(false)}
                          onEnded={() => setActiveVideoPlaying(false)}
                          onClick={() => {
                            if (activeVideoRef.current?.paused) {
                              playActiveVideo();
                              return;
                            }
                            activeVideoRef.current?.pause();
                          }}>
                          <track kind="captions" label="无字幕" />
                        </video>
                        {!activeVideoPlaying ? (
                          <button
                            type="button"
                            className="hcai-video-play-button"
                            aria-label="播放视频"
                            onClick={playActiveVideo}>
                            <Icon name="play-fill" />
                          </button>
                        ) : null}
                      </div>
                      <div className="hcai-preview-tile-meta">
                        <div>
                          <strong>{activeTask.prompt}</strong>
                          <span>
                            {activeTask.size} · {activeTask.seconds} 秒 ·{' '}
                            {activeTask.quality}
                          </span>
                        </div>
                        <button
                          type="button"
                          title="放大预览"
                          aria-label="放大预览"
                          onClick={() =>
                            openVideoPreview(
                              activeTask,
                              activeVideoURL,
                              activeTask.videoURL,
                            )
                          }>
                          <Icon name="arrows-fullscreen" />
                        </button>
                        <button
                          type="button"
                          title="下载视频"
                          aria-label="下载视频"
                          onClick={() =>
                            downloadVideo(
                              resolveVideoURL(activeTask.videoURL),
                              activeTask,
                            )
                          }>
                          <Icon name="download" />
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <Icon name="camera-reels" />
                      <span>视频加载中</span>
                    </>
                  )
                ) : activeTask.status === 'failed' ? (
                  <>
                    <Icon name="exclamation-triangle" />
                    <span>生成失败</span>
                  </>
                ) : (
                  <>
                    <Icon name="camera-reels" />
                    <span>{activeTask.ratio}</span>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="hcai-preview-empty">
              <Icon name="camera-reels" />
              <span>生成结果会显示在这里</span>
            </div>
          )}
        </div>

        {renderTaskPanel({})}
      </section>
      {previewVideo ? (
        <div
          className="hcai-image-lightbox hcai-video-lightbox"
          role="dialog"
          aria-modal="true">
          <button
            type="button"
            className="hcai-image-lightbox-backdrop"
            aria-label="关闭预览"
            onClick={() => setPreviewVideo(null)}
          />
          <div className="hcai-image-lightbox-bar">
            <div>
              <strong>{previewVideo.task.prompt}</strong>
              <span>
                {previewVideo.task.size} · {previewVideo.task.seconds} 秒 ·{' '}
                {previewVideo.task.quality}
              </span>
            </div>
            <button
              type="button"
              title="下载"
              aria-label="下载"
              onClick={(evt) => {
                evt.stopPropagation();
                downloadVideo(
                  resolveVideoURL(previewVideo.rawURL),
                  previewVideo.task,
                );
              }}>
              <Icon name="download" />
            </button>
            <button
              type="button"
              title="关闭"
              aria-label="关闭"
              onClick={(evt) => {
                evt.stopPropagation();
                setPreviewVideo(null);
              }}>
              <Icon name="x-lg" />
            </button>
          </div>
          <div className="hcai-lightbox-video-shell">
            <video
              ref={previewVideoRef}
              src={previewVideo.src}
              playsInline
              onPlay={() => setPreviewVideoPlaying(true)}
              onPause={() => setPreviewVideoPlaying(false)}
              onEnded={() => setPreviewVideoPlaying(false)}
              onClick={(evt) => {
                evt.stopPropagation();
                if (previewVideoRef.current?.paused) {
                  playPreviewVideo();
                  return;
                }
                previewVideoRef.current?.pause();
              }}>
              <track kind="captions" label="无字幕" />
            </video>
            {!previewVideoPlaying ? (
              <button
                type="button"
                className="hcai-video-play-button"
                aria-label="播放视频"
                onClick={(evt) => {
                  evt.stopPropagation();
                  playPreviewVideo();
                }}>
                <Icon name="play-fill" />
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default VideoGenerationWorkspace;
