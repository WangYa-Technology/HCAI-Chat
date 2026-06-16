import { useEffect } from 'react';
import { hasAnyImageAgentThinkingModel, initStore, useStore } from './store';
import {
  buildSettingsFromUrlParams,
  clearUrlSettingParams,
  hasUrlSettingParams,
} from './lib/urlSettings';
import { mergeImportedSettings } from './lib/apiProfiles';
import {
  getCustomProviderConfigUrl,
  loadCustomProviderSettingsFromUrl,
} from './lib/customProviderConfigUrl';
import {
  consumePendingImagePrompt,
  PENDING_IMAGE_PROMPT_EVENT,
} from './pendingPrompt';
import Header from './components/Header';
import PlaygroundTopbar from './components/PlaygroundTopbar';
import TaskGrid from './components/TaskGrid';
import AgentWorkspace from './components/AgentWorkspace';
import AgentConversationTaskPanel from './components/AgentConversationTaskPanel';
import GalleryTaskPanel from './components/GalleryTaskPanel';
import InputBar from './components/InputBar';
import DetailModal from './components/DetailModal';
import Lightbox from './components/Lightbox';
import SettingsModal from './components/SettingsModal';
import ConfirmDialog from './components/ConfirmDialog';
import Toast from './components/Toast';
import MaskEditorModal from './components/MaskEditorModal';
import ImageContextMenu from './components/ImageContextMenu';
import SupportPromptModal from './components/SupportPromptModal';
import {
  FavoriteCollectionPickerModal,
  FavoriteCollectionsView,
  ManageCollectionsModal,
} from './components/FavoriteCollections';
import { useGlobalClickSuppression } from './lib/clickSuppression';

let customProviderConfigUrlImportStarted = false;
let imageGenerationInitPromise: Promise<void> | null = null;

function ensureImageGenerationInitialized() {
  if (!imageGenerationInitPromise) {
    imageGenerationInitPromise = initStore().catch((error) => {
      imageGenerationInitPromise = null;
      throw error;
    });
  }
  return imageGenerationInitPromise;
}

interface AppProps {
  embedded?: boolean;
}

export default function App({ embedded = false }: AppProps) {
  const setSettings = useStore((s) => s.setSettings);
  const setPrompt = useStore((s) => s.setPrompt);
  const loadSystemImageModels = useStore((s) => s.loadSystemImageModels);
  const loadSystemImageGenerations = useStore(
    (s) => s.loadSystemImageGenerations,
  );
  const appMode = useStore((s) => s.appMode);
  const setAppMode = useStore((s) => s.setAppMode);
  const systemImageModels = useStore((s) => s.systemImageModels);
  const systemImageModelsLoaded = useStore((s) => s.systemImageModelsLoaded);
  const systemImageModelsLoading = useStore((s) => s.systemImageModelsLoading);
  const hasRunningGalleryTasks = useStore((s) =>
    s.tasks.some(
      (task) =>
        (task.sourceMode ?? 'gallery') === 'gallery' &&
        task.status === 'running',
    ),
  );
  const filterFavorite = useStore((s) => s.filterFavorite);
  const activeFavoriteCollectionId = useStore(
    (s) => s.activeFavoriteCollectionId,
  );
  useGlobalClickSuppression();

  useEffect(() => {
    const applyPrompt = (prompt: string) => {
      const trimmedPrompt = prompt.trim();
      if (!trimmedPrompt) {
        return;
      }
      setAppMode('gallery');
      setPrompt(trimmedPrompt);
    };
    const handlePromptEvent = (evt: Event) => {
      const prompt =
        evt instanceof CustomEvent && typeof evt.detail?.prompt === 'string'
          ? evt.detail.prompt
          : '';
      applyPrompt(prompt);
    };

    window.addEventListener(PENDING_IMAGE_PROMPT_EVENT, handlePromptEvent);
    return () => {
      window.removeEventListener(PENDING_IMAGE_PROMPT_EVENT, handlePromptEvent);
    };
  }, [setAppMode, setPrompt]);

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const nextSettings = buildSettingsFromUrlParams(
      useStore.getState().settings,
      searchParams,
    );

    setSettings(nextSettings);

    if (hasUrlSettingParams(searchParams)) {
      clearUrlSettingParams(searchParams);

      const nextSearch = searchParams.toString();
      const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}${window.location.hash}`;
      window.history.replaceState(null, '', nextUrl);
    }

    const customProviderConfigUrl = getCustomProviderConfigUrl();
    if (customProviderConfigUrl && !customProviderConfigUrlImportStarted) {
      customProviderConfigUrlImportStarted = true;
      void loadCustomProviderSettingsFromUrl(customProviderConfigUrl)
        .then((importedSettings) => {
          if (!importedSettings) return;
          const state = useStore.getState();
          state.setSettings(
            mergeImportedSettings(state.settings, importedSettings),
          );
        })
        .catch((error) => {
          console.warn('Failed to import custom provider config URL:', error);
        });
    }

    let cancelled = false;
    void (async () => {
      const initPromise = ensureImageGenerationInitialized();
      const historyPromise = loadSystemImageGenerations();
      await initPromise;
      const pendingPrompt = consumePendingImagePrompt();
      if (!cancelled && pendingPrompt) {
        setAppMode('gallery');
        setPrompt(pendingPrompt);
      }
      if (!cancelled) await historyPromise;
    })();
    void loadSystemImageModels();

    return () => {
      cancelled = true;
    };
  }, [
    loadSystemImageGenerations,
    loadSystemImageModels,
    setAppMode,
    setPrompt,
    setSettings,
  ]);

  useEffect(() => {
    if (
      appMode === 'agent' &&
      systemImageModelsLoaded &&
      !systemImageModelsLoading &&
      !hasAnyImageAgentThinkingModel(systemImageModels)
    ) {
      setAppMode('gallery');
    }
  }, [
    appMode,
    setAppMode,
    systemImageModels,
    systemImageModelsLoaded,
    systemImageModelsLoading,
  ]);

  useEffect(() => {
    if (!hasRunningGalleryTasks) return;

    let cancelled = false;
    const refreshGenerations = () => {
      if (cancelled || document.visibilityState === 'hidden') return;
      void loadSystemImageGenerations({ silent: true });
    };

    const intervalId = window.setInterval(refreshGenerations, 3000);
    window.addEventListener('focus', refreshGenerations);
    document.addEventListener('visibilitychange', refreshGenerations);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      window.removeEventListener('focus', refreshGenerations);
      document.removeEventListener('visibilitychange', refreshGenerations);
    };
  }, [hasRunningGalleryTasks, loadSystemImageGenerations]);

  useEffect(() => {
    const preventPageImageDrag = (e: DragEvent) => {
      if ((e.target as HTMLElement | null)?.closest('img')) {
        e.preventDefault();
      }
    };

    document.addEventListener('dragstart', preventPageImageDrag);
    return () =>
      document.removeEventListener('dragstart', preventPageImageDrag);
  }, []);

  return (
    <>
      {!embedded && <Header />}
      {appMode === 'agent' ? (
        <main
          data-agent-workspace
          className="flex min-h-[calc(100vh-100px)] flex-col relative overflow-visible transition-all duration-300">
          <div className="safe-area-x max-w-7xl mx-auto">
            <PlaygroundTopbar mode="agent" />
            <AgentWorkspace />
          </div>
        </main>
      ) : (
        <main data-home-main data-drag-select-surface className="pb-48">
          <div className="safe-area-x max-w-7xl mx-auto">
            <PlaygroundTopbar mode="gallery" />
            {filterFavorite && !activeFavoriteCollectionId ? (
              <FavoriteCollectionsView />
            ) : (
              <TaskGrid />
            )}
          </div>
        </main>
      )}
      <InputBar />
      <GalleryTaskPanel />
      <AgentConversationTaskPanel />
      <DetailModal />
      <Lightbox />
      <SettingsModal />
      <ConfirmDialog />
      <SupportPromptModal />
      <FavoriteCollectionPickerModal />
      <ManageCollectionsModal />
      <Toast />
      <MaskEditorModal />
      <ImageContextMenu />
    </>
  );
}
